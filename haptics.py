#!/usr/bin/env python3
"""
haptics.py: live sound -> 3 vibration motors on a Raspberry Pi.

Listens to the USB mic, splits what it hears into low / mid / high bands
and drives one vibration motor per band, so you feel the kick, the snare
and the hats. Separate program from drum_pi.py; nothing is shared.

The motor patterns follow the FELT-MX framework's layers, in its priority
order: Texture (the 3 bands stand in for 3 voices), Rhythm and timing
(accented hits first, then the other hits), Dynamics (5 strength levels) and
Tonality (sustained sounds give longer pulses). Meter and Meaning are not
represented: meter is usually implied by the rhythm, and meaning needs text
or visuals, which this device does not have. HAPTIC_LAYERS picks the layers.

Run modes:
    python3 haptics.py              # on the Pi: USB mic in, PWM motors out
    python3 haptics.py --sim        # laptop: mic in, motor levels drawn as bars
    python3 haptics.py verify       # headless self-test, no audio hardware
    python3 haptics.py test FILE    # run a .wav file through and print the bars

Only numpy is required for verify/test. The device also needs sounddevice,
plus gpiozero (with lgpio) on the Pi.
"""

import argparse
import os
import queue
import sys
import time
import wave

import numpy as np

# ─── CONFIG ───────────────────────────────────────────────
SAMPLE_RATE = 16000        # 8 kHz of bandwidth is plenty for touch
BLOCK_SEC   = 0.016        # analysis step: one motor update every 16 ms
FFT_SEC     = 0.032        # FFT window (rounded up to a power of 2)

#             name    low Hz  high Hz   felt as
BANDS = [("low",     30,    200),     # kick, bass
         ("mid",    200,   2000),     # snare, toms, voice
         ("high",  2000,   7500)]     # hi-hat, cymbal

# Auto-scaling: each band's level is judged against its own recent quiet and
# loud, so there are no fixed loudness thresholds to tune per room.
FLOOR_RISE_SEC   = 3.0     # noise floor follows steady sound up with this time constant
FLOOR_FALL_DB_S  = 30.0    # ...and drops quickly when it gets quieter
PEAK_DECAY_DB_S  = 3.0     # loudest-recent level relaxes this fast
MIN_RANGE_DB     = 30.0    # peak is never closer than this to the floor (keeps noise out of the range)
RANGE_DB         = 20.0    # motor goes from off to full over the top 20 dB below the peak
GATE_DB          = 8.0     # must be this far above the floor to count at all
ABS_FLOOR_DB     = -80.0   # true silence: never treat less than this as signal

# Transients: a sudden jump in a band gives a full-strength kick to the motor
# even if the band is not loud overall. This is what makes hits feel crisp.
FLUX_DB          = 6.0     # jump (vs. the band's recent average) that starts a boost
FLUX_FULL_DB     = 15.0    # jump that gives a full-strength boost

ATTACK_SEC  = 0.005        # envelope rise (effectively instant at 16 ms blocks)
RELEASE_SEC = 0.120        # envelope fall: long enough to feel, short enough to separate hits

# Motors. Coin ERM motors don't start below ~30 % duty, and 3 V motors fed
# ~4 V through the ULN2003 must be capped so they don't overheat.
MOTOR_PINS   = [4, 12, 7]    # BCM: low, mid, high (physical pins 7, 32, 26). Not 21: drum_pi uses it for the I2S amp
PWM_FREQ     = 500           # software PWM; hardware PWM on a Pi 3 clashes with the audio jack
ON_LEVEL     = 0.05          # envelope below this = motor off
MIN_DUTY     = 0.35
MAX_DUTY     = float(os.environ.get("HAPTIC_MAX_DUTY", 0.75))
BAND_GAIN    = [1.0, 1.0, 1.0]   # per-motor trim if one feels stronger than the others

# ─── FELT-MX LAYERS ───────────────────────────────────────
# Each detected hit becomes a discrete pulse instead of the motor following
# loudness continuously: with few motors, overlapping layers blur into one
# constant buzz, so the framework keeps the most important layers distinct.
#   emphasis  accented hits (louder than this band's recent hits): long, strong pulse
#   rhythm    every other hit: short, softer pulse, dropped if the motor is busy
#   dynamics  pulse strength follows loudness in 5 steps (all a hand can tell apart)
#   timbre    a sound that keeps ringing keeps its pulse going (legato vs staccato)
#   follow    the old continuous loudness mode, under the pulses (off by default)
LAYERS = set(os.environ.get("HAPTIC_LAYERS", "emphasis,rhythm,dynamics,timbre").split(","))
MOTORS_ON = [c != "0" for c in os.environ.get("HAPTIC_MOTORS", "111")[:len(BANDS)]]  # texture: which voices

ACCENT_PULSE_SEC = 0.100   # emphasis pulse
RHYTHM_PULSE_SEC = 0.048   # other hits (3 blocks: about the shortest an ERM motor shows clearly)
SUSTAIN_MAX_SEC  = 0.300   # timbre: longest a ringing sound can stretch a pulse
MIN_OFF_SEC      = 0.040   # gap kept before the next expected hit, so fast hits stay separate
SUSTAIN_LEVEL    = 0.5     # ...while the band stays this loud (0..1)
ACCENT_GAP_SEC   = 0.090   # an accent may start this soon after the last pulse began
RHYTHM_GAP_SEC   = 0.150   # other hits need more room, so fast rolls don't blur
RECENT_HITS_SEC  = 4.0     # accents are judged against this band's hits in the last 4 s
ACCENT_DB        = 3.0     # within 3 dB of the loudest of those = accented
DYN_LEVELS       = 5
BLEED_DB         = 18.0    # a hit this much quieter than one in another band at the same moment is leakage

AUDIO_IN_DEVICE = os.environ.get("HAPTIC_IN") or None
REOPEN_SEC      = 2.0        # mic busy or unplugged: try again this often


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── ANALYSIS ─────────────────────────────────────────────
def _coef(tau, dt):
    """One-pole smoothing coefficient for time constant tau at step dt."""
    return 1.0 - np.exp(-dt / tau) if tau > 0 else 1.0


class BandFollower:
    """
    Audio blocks in, three motor levels (0..1) out.

    Per block: FFT of the last FFT_SEC of audio -> energy in each band (dB)
    -> scaled 0..1 between that band's tracked noise floor and recent peak
    -> boosted on sudden jumps -> fast-attack / slow-release envelope.
    """

    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.block = max(1, int(round(BLOCK_SEC * sr)))
        self.n_fft = 1 << int(np.ceil(np.log2(FFT_SEC * sr)))
        self.window = np.hanning(self.n_fft).astype(np.float32)
        self.ring = np.zeros(self.n_fft, dtype=np.float32)
        freqs = np.fft.rfftfreq(self.n_fft, 1 / sr)
        self.masks = [(freqs >= lo) & (freqs < min(hi, sr / 2)) for _, lo, hi in BANDS]
        # Hann window power, so dB values don't depend on the FFT size.
        self.norm = float((self.window ** 2).sum()) * self.n_fft / 2
        dt = self.block / sr
        self.dt = dt
        self.a_att = _coef(ATTACK_SEC, dt)
        self.a_rel = _coef(RELEASE_SEC, dt)
        self.a_avg = _coef(0.150, dt)            # band's recent average, for transients
        self.a_floor = _coef(FLOOR_RISE_SEC, dt)
        n = len(BANDS)
        self.floor = np.full(n, np.nan)
        self.peak = np.full(n, np.nan)
        self.avg = np.full(n, np.nan)
        self.env = np.zeros(n)
        self._pending = np.zeros(0, dtype=np.float32)

    def band_db(self):
        spec = np.abs(np.fft.rfft(self.ring * self.window)) ** 2 / self.norm
        return np.array([10 * np.log10(spec[m].sum() + 1e-12) for m in self.masks])

    def step(self, block):
        """Advance by one block of exactly self.block samples. Returns env (0..1 per band)."""
        n = len(block)
        self.ring[:-n] = self.ring[n:]
        self.ring[-n:] = block
        db = np.maximum(self.band_db(), ABS_FLOOR_DB)
        dt = self.dt

        first = np.isnan(self.floor)
        self.floor[first] = db[first]
        self.peak[first] = db[first] + MIN_RANGE_DB
        self.avg[first] = db[first]

        self.floor = np.where(db < self.floor,
                              np.maximum(db, self.floor - FLOOR_FALL_DB_S * dt),
                              self.floor + self.a_floor * (db - self.floor))
        self.peak = np.maximum(db, self.peak - PEAK_DECAY_DB_S * dt)
        self.peak = np.maximum(self.peak, self.floor + MIN_RANGE_DB)

        level = np.clip((db - self.peak + RANGE_DB) / RANGE_DB, 0.0, 1.0)
        level[db - self.floor < GATE_DB] = 0.0

        jump = db - self.avg
        self.last = {"db": db, "level": level.copy(), "jump": jump}
        boost = np.clip((jump - FLUX_DB) / (FLUX_FULL_DB - FLUX_DB), 0.0, 1.0)
        boost[level == 0] = 0.0      # a jump in noise, or bleed from a much louder band, isn't a hit
        self.avg += self.a_avg * (db - self.avg)

        target = np.maximum(level, boost)
        a = np.where(target > self.env, self.a_att, self.a_rel)
        self.env += a * (target - self.env)
        return self.env.copy()

    def feed(self, audio):
        """Any number of samples in; one env per completed block out (a list)."""
        x = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).ravel()])
        out = []
        i = 0
        while i + self.block <= len(x):
            out.append(self.step(x[i:i + self.block]))
            i += self.block
        self._pending = x[i:]
        return out


class FeltMx:
    """
    Per-band analysis in, motor duty out, one block at a time.

    A hit is a rising jump of FLUX_DB in a band that is above its gate. It is
    judged one block later (16 ms), once its peak is known: dropped if it is
    leakage from a much louder hit in another band, otherwise compared with
    this band's hits of the last few seconds to decide accent or not and
    which of the 5 dynamics levels it gets. Then its pulse starts.
    """

    def __init__(self, dt, layers=None, motors_on=None):
        self.dt = dt
        self.layers = LAYERS if layers is None else set(layers)
        self.on = MOTORS_ON if motors_on is None else list(motors_on)
        n = len(BANDS)
        self.t = 0.0
        self.armed = np.ones(n, dtype=bool)   # ready for a new hit (the last jump has settled)
        self.prev_db = np.full(n, ABS_FLOOR_DB)
        self.rose = np.zeros(n, dtype=bool)   # a hit started in the previous block
        self.start = np.full(n, -1e9)         # when the current pulse started
        self.until = np.zeros(n)              # when it ends
        self.duty = np.zeros(n)
        self.longest = np.zeros(n)            # cap for a ringing sound's pulse
        self.hits = [[] for _ in range(n)]    # (time, peak dB) of recent hits per band

    def _pulse(self, b, peak):
        t = self.t
        self.hits[b] = [(ht, d) for (ht, d) in self.hits[b] if t - ht <= RECENT_HITS_SEC]
        self.hits[b].append((t, peak))
        rel = peak - max(d for _, d in self.hits[b])          # 0 = as loud as the loudest
        accent = rel >= -ACCENT_DB
        since = t - self.start[b]
        if accent and "emphasis" in self.layers and since >= ACCENT_GAP_SEC:
            length = ACCENT_PULSE_SEC
        elif "rhythm" in self.layers and since >= RHYTHM_GAP_SEC and t >= self.until[b]:
            length, accent = RHYTHM_PULSE_SEC, False
        else:
            return                                             # no room, or layer off
        longest = SUSTAIN_MAX_SEC
        times = [ht for ht, _ in self.hits[b]]
        if len(times) >= 2:                                    # leave a felt gap before the next hit
            ioi = float(np.median(np.diff(times)))
            longest = max(RHYTHM_PULSE_SEC, min(longest, ioi - MIN_OFF_SEC - self.dt))  # dt: pulses end on block edges
        length = min(length, longest)
        if "dynamics" in self.layers:                          # 5 levels, 4 dB apart below the accent band
            step = int(np.clip(DYN_LEVELS + np.floor((rel + ACCENT_DB) / 4.0), 1, DYN_LEVELS))
        else:
            step = DYN_LEVELS if accent else 3
        self.start[b], self.until[b], self.longest[b] = t, t + length, longest
        self.duty[b] = MIN_DUTY + (MAX_DUTY - MIN_DUTY) * (step - 1) / (DYN_LEVELS - 1)

    def step(self, a, follow_env=None):
        """a = BandFollower.last. Returns duty per motor for this block."""
        self.t += self.dt
        t = self.t
        db, level, jump = a["db"], a["level"], a["jump"]
        rising = (jump >= FLUX_DB) & self.armed & (level > 0)
        self.armed = (self.armed & ~rising) | (jump < FLUX_DB)
        peak = np.maximum(db, self.prev_db)                    # each band over the last 2 blocks
        hitting = rising | self.rose
        for b in np.nonzero(self.rose)[0]:                     # judge last block's hits now
            if not self.on[b]:
                continue
            # Leakage: far quieter than a hit in another band at the same moment
            # (a kick's click showing up faintly in the mid band).
            if any(hitting[c] and peak[c] - peak[b] >= BLEED_DB
                   for c in range(len(BANDS)) if c != b):
                continue
            self._pulse(b, peak[b])
        for b in range(len(BANDS)):                            # timbre: a ringing sound keeps its pulse
            if ("timbre" in self.layers and self.on[b] and self.until[b] - self.dt <= t < self.until[b]
                    and level[b] >= SUSTAIN_LEVEL):
                self.until[b] = min(t + 1.5 * self.dt, self.start[b] + self.longest[b])
        self.rose = rising
        self.prev_db = db.copy()
        out = np.where(t < self.until, self.duty, 0.0) * np.array(self.on, dtype=float)
        if "follow" in self.layers and follow_env is not None:
            out = np.maximum(out, to_duty(follow_env) * np.array(self.on, dtype=float))
        return out


class Haptics:
    """Audio in, one duty array per 16 ms block out."""

    def __init__(self, sr=SAMPLE_RATE, layers=None, motors_on=None):
        self.follower = BandFollower(sr)
        self.block = self.follower.block
        self.n_fft = self.follower.n_fft
        self.felt = FeltMx(self.follower.dt, layers, motors_on)
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, audio):
        x = np.concatenate([self._pending, np.asarray(audio, dtype=np.float32).ravel()])
        out, i = [], 0
        while i + self.block <= len(x):
            env = self.follower.step(x[i:i + self.block])
            out.append(np.asarray(self.felt.step(self.follower.last, env)) * np.asarray(BAND_GAIN))
            i += self.block
        self._pending = x[i:]
        return out


def to_duty(env):
    """Envelope (0..1 per band) -> PWM duty per motor."""
    env = np.asarray(env) * np.asarray(BAND_GAIN)
    duty = MIN_DUTY + (MAX_DUTY - MIN_DUTY) * np.clip(env, 0.0, 1.0)
    return np.where(env < ON_LEVEL, 0.0, duty)


# ─── MOTORS ───────────────────────────────────────────────
class GpioMotors:
    def __init__(self):
        from gpiozero import PWMOutputDevice
        self.m = [PWMOutputDevice(p, frequency=PWM_FREQ, initial_value=0) for p in MOTOR_PINS]
        self.last = [None] * len(self.m)

    def set(self, duty):
        for i, d in enumerate(duty):
            d = round(float(d), 2)               # don't hammer lgpio with tiny changes
            if d != self.last[i]:
                self.m[i].value = d
                self.last[i] = d

    def off(self):
        for m in self.m:
            m.value = 0


class BarMotors:
    """Stand-in for the motors: draws one line of bars per update."""

    def __init__(self, every=1):
        self.every, self.n = every, 0

    def set(self, duty):
        self.n += 1
        if self.n % self.every:
            return
        bars = "  ".join(f"{name[0].upper()} {'#' * int(d * 20):<20}"
                         for (name, _, _), d in zip(BANDS, duty))
        sys.stdout.write("\r" + bars)
        sys.stdout.flush()

    def off(self):
        sys.stdout.write("\n")


# ─── LIVE LOOP ────────────────────────────────────────────
def open_rate(sd):
    """16 kHz if the mic supports it, else its own default rate."""
    try:
        sd.check_input_settings(device=AUDIO_IN_DEVICE, samplerate=SAMPLE_RATE, channels=1)
        return SAMPLE_RATE
    except Exception:
        return int(sd.query_devices(AUDIO_IN_DEVICE, "input")["default_samplerate"])


def run_live(motors):
    import sounddevice as sd
    try:
        while True:
            try:
                listen(sd, motors)
            except Exception as e:                # mic busy (drum_pi recording?) or unplugged
                motors.off()
                log(f"[mic] {e}; retrying in {REOPEN_SEC:.0f} s")
                time.sleep(REOPEN_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        motors.off()


def listen(sd, motors):
    rate = open_rate(sd)
    follower = Haptics(rate)
    blocks = queue.Queue(maxsize=32)

    def cb(indata, frames, time_info, status):
        try:
            blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass                                  # falling behind: drop, never block audio

    with sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                        blocksize=follower.block, device=AUDIO_IN_DEVICE, callback=cb):
        log(f"[mic] listening at {rate} Hz, {follower.block}-sample blocks, "
            f"FFT {follower.n_fft}")
        while True:
            x = blocks.get(timeout=1.0)           # no audio for 1 s = stream died -> reopen
            while blocks.qsize() > 2:             # behind by >2 blocks: skip to now
                follower.feed(x)
                x = blocks.get_nowait()
            duties = follower.feed(x)
            if duties:
                motors.set(np.clip(duties[-1], 0.0, MAX_DUTY))


# ─── FILE MODE / SELF-TEST ────────────────────────────────
def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    else:
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(axis=1), sr
    return x.reshape(-1, ch).mean(axis=1), sr


def run_file(path):
    x, sr = read_wav(path)
    f = Haptics(sr)
    every = max(1, int(0.05 / (f.block / sr)))     # one printed line per ~50 ms
    for k, d in enumerate(f.feed(x)):
        if k % every == 0:
            t = k * f.block / sr
            print(f"{t:6.2f}s  " + "  ".join(
                f"{n[0].upper()} {'#' * int(v * 20):<20}" for (n, _, _), v in zip(BANDS, d)))


def verify():
    sr = SAMPLE_RATE
    rng = np.random.default_rng(1)
    ok = [True]

    def check(name, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok[0] &= bool(cond)

    def tone(freq, sec, amp=0.5):
        t = np.arange(int(sec * sr)) / sr
        return (amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * 6)).astype(np.float32)

    def hiss(sec, amp=0.3, decay=25):
        n = int(sec * sr)
        x = rng.standard_normal(n).astype(np.float32)
        x = x - np.convolve(x, np.ones(4) / 4, mode="same")  # crude high-pass
        return (amp * x * np.exp(-np.arange(n) / sr * decay)).astype(np.float32)

    def run(signal, lead=1.0):
        f = BandFollower(sr)
        quiet = 0.001 * rng.standard_normal(int(lead * sr)).astype(np.float32)
        envs = np.array(f.feed(np.concatenate([quiet, signal])))
        return envs, int(lead * sr) // f.block, f

    print("haptics self-test")
    for name, sig, band in [("kick (60 Hz)", tone(60, 0.4), 0),
                            ("snare body (800 Hz)", tone(800, 0.3), 1),
                            ("hi-hat (hiss)", hiss(0.2), 2)]:
        envs, start, f = run(sig)
        peak = envs[start:].max(axis=0)
        check(f"{name}: motor {band + 1} strongest", int(np.argmax(peak)) == band)
        check(f"{name}: motor {band + 1} turns on", to_duty(peak)[band] > 0)
        rise = int(np.argmax(envs[start:, band] > 0.5))
        check(f"{name}: on within 2 blocks ({rise * f.block / sr * 1000:.0f} ms)", rise <= 2)

    envs, start, _ = run(np.zeros(sr, dtype=np.float32))
    check("silence: all motors off", not to_duty(envs[-1]).any())

    hum = (0.2 * np.sin(2 * np.pi * 100 * np.arange(12 * sr) / sr)).astype(np.float32)
    envs, start, _ = run(hum)
    check("steady hum fades out (floor adapts)", to_duty(envs[-1])[0] == 0)

    beat = np.zeros(2 * sr, dtype=np.float32)
    for k in range(4):                                   # 4 kicks, 0.5 s apart
        i = k * sr // 2
        beat[i:i + int(0.4 * sr)] += tone(60, 0.4)
    envs, start, f = run(beat)
    low = envs[start:, 0]
    gap = low[int(0.45 * sr) // f.block]                 # just before kick 2
    check(f"separate kicks felt separately (env {gap:.2f} before next hit)", gap < 0.4)

    check("duty within motor limits", all(0 <= d <= MAX_DUTY for d in to_duty([1, 0.5, 0.01])))

    # FELT-MX layers
    def place(events, total):
        y = np.zeros(int(total * sr), dtype=np.float32)
        for t0, sig in events:
            i = int(t0 * sr)
            n = min(len(sig), len(y) - i)
            y[i:i + n] += sig[:n]
        return y

    def faded(x):
        x = x.copy()
        x[-160:] *= np.linspace(1, 0, 160, dtype=np.float32)
        return x

    def pulses(signal, **kw):
        hp = Haptics(sr, **kw)
        lead = int(sr)
        d = np.array(hp.feed(np.concatenate([0.001 * rng.standard_normal(lead).astype(np.float32),
                                             signal])))[lead // hp.block:]
        dt, out = hp.block / sr, []
        for b in range(len(BANDS)):
            on = d[:, b] > 0
            starts = np.nonzero(on & ~np.r_[False, on[:-1]])[0]
            ps = []
            for st in starts:
                e = st
                while e < len(on) and on[e]:
                    e += 1
                ps.append((st * dt, (e - st) * dt, float(d[st:e, b].max())))
            out.append(ps)
        return out

    kicks = pulses(place([(k * 0.5, faded(tone(60, 0.4))) for k in range(4)], 2.2))
    check(f"4 kicks -> 4 pulses on motor 1 only (got {[len(p) for p in kicks]})",
          [len(p) for p in kicks] == [4, 0, 0])
    alt = pulses(place([(k * 0.5, faded(tone(60, 0.4, 0.5 if k % 2 == 0 else 0.15)))
                        for k in range(8)], 4.2))[0]
    loud, soft = alt[0::2], alt[1::2]
    check("loud kicks are accents: longer and stronger than soft ones",
          len(alt) == 8 and min(p[1] for p in loud) > max(p[1] for p in soft)
          and min(p[2] for p in loud) > max(p[2] for p in soft))
    levels = pulses(place([(k * 0.5, faded(tone(60, 0.4, amp)))
                           for k, amp in enumerate([0.5, 0.4, 0.3, 0.2, 0.12, 0.08])], 3.2))[0]
    check(f"dynamics: quieter hits step down through 5 levels "
          f"({[round(p[2], 2) for p in levels]})",
          len({round(p[2], 2) for p in levels}) == DYN_LEVELS
          and all(a >= b for a, b in zip([p[2] for p in levels], [p[2] for p in levels][1:])))
    hats = pulses(place([(k * 0.125, hiss(0.08)) for k in range(16)], 2.2))[2]
    gaps = [b[0] - (a[0] + a[1]) for a, b in zip(hats, hats[1:])]
    # The first pulse can't know the tempo yet, so the gap is only promised from the 2nd hit on.
    check(f"fast hi-hats: every hit felt, with a gap between once the tempo is known "
          f"({len(hats)} pulses, min gap {min(gaps[1:]) * 1000:.0f} ms)",
          len(hats) == 16 and min(gaps[1:]) >= 0.03)
    ring = pulses(hiss(1.5, 0.3, decay=2))[2]
    short = pulses(hiss(0.08))[2]
    check("timbre: a ringing cymbal gives a longer pulse than a short hat",
          ring and short and ring[0][1] > short[0][1] + 0.1)
    plain = pulses(place([(k * 0.5, faded(tone(60, 0.4, 0.5 if k % 2 == 0 else 0.15)))
                          for k in range(8)], 4.2), layers={"rhythm"})[0]
    check("layers can be switched off (rhythm only: all pulses the same)",
          len(plain) == 8 and len({round(p[2], 2) for p in plain}) == 1)
    muted = pulses(place([(k * 0.5, faded(tone(60, 0.4))) for k in range(4)], 2.2),
                   motors_on=[False, True, True])
    check("texture: a motor can be turned off", not muted[0])
    print("ALL PASS" if ok[0] else "SOME CHECKS FAILED")
    return 0 if ok[0] else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="run", choices=["run", "verify", "test"])
    ap.add_argument("file", nargs="?")
    ap.add_argument("--sim", action="store_true", help="draw motor bars instead of using GPIO")
    args = ap.parse_args()
    if args.mode == "verify":
        sys.exit(verify())
    if args.mode == "test":
        if not args.file:
            ap.error("test needs a .wav file")
        run_file(args.file)
        return
    motors = BarMotors(every=3) if args.sim else GpioMotors()
    log(f"[haptics] motors on BCM {MOTOR_PINS}" if not args.sim else "[haptics] sim mode")
    run_live(motors)


if __name__ == "__main__":
    main()
