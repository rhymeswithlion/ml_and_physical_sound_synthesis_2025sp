import hashlib
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path

import jinja2
from loguru import logger

TMP_DIR = Path("/tmp/pychuck")


class ChuckShred:
    def __init__(self, code, **params):
        def _remove_comments(c):
            return "\n".join(
                [line for line in c.split("\n") if not line.strip().startswith("#")]
            )

        code = _remove_comments(code)
        self.code = jinja2.Template(code).render(**params)

        self.path = TMP_DIR / (self.sha256 + ".ck")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.code)

    @cached_property
    def sha256(self):
        import hashlib

        return hashlib.sha256(self.code.encode()).hexdigest()

    @cached_property
    def friendly_name(self):
        import randomname

        return randomname.get_name(seed=self.sha256).replace("-", "_")

    def run(self, silent=False, max_seconds=5):
        cmd = ["chuck", self.path.as_posix()]
        if silent:
            cmd.append("--silent")

        # Run for a maximum of max_seconds
        proc = subprocess.Popen(
            cmd,
            cwd=TMP_DIR.as_posix(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # print("starting")

        try:
            # print("waiting")
            proc.wait(timeout=max_seconds)
            # print("done")
        except subprocess.TimeoutExpired:
            # print("timed out")
            proc.terminate()
            proc.wait()

        stdout = proc.stdout.read()
        stderr = proc.stderr.read()

        for line in stdout.splitlines():
            logger.info(line.decode())

        for line in stderr.splitlines():
            logger.info(str(line.decode()))

        return stdout, stderr


def write_wave(path, samples, sr=44100):
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


class AudioClip:
    def __init__(self, samples, sr=44100):
        self.samples = samples
        self.sr = sr

        # write to file
        self.path = TMP_DIR / f"{self.sha256}.wav"
        write_wave(self.path, self.samples, self.sr)

    @cached_property
    def data(self):
        return self.samples.tobytes()

    @cached_property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()

    def display(self):
        from audio_utils import play_audio

        play_audio(self.samples, sr=self.sr, auto_play=True)

    def to_file(self, path):
        write_wave(path, self.samples, self.sr)


def shreds_to_audio(*shreds, num_ms=1000):
    from audio_utils import load_audio

    with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
        code = jinja2.Template(
            """
            // iterate over each shred
            {% for shred in shreds %}
            Machine.add( "{{ shred.path.as_posix() }}" ) => int {{shred.friendly_name}};
            {% endfor %}

            dac => WvOut waveOut => blackhole;
            "{{tf.name}}"=>waveOut.wavFilename;

            {{ num_ms }}::ms => now;

            // remove the shred
            {% for shred in shreds %}
            Machine.remove( {{ shred.friendly_name }} );
            {% endfor %}

            waveOut.closeFile;

            """
        ).render(shreds=shreds, num_ms=num_ms, tf=tf)
        # return code

        machine_shred = ChuckShred(code)

        machine_shred.run(silent=True)

        return AudioClip(load_audio(tf.name))
