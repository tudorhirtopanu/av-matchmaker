import sys
import wave
from collections import namedtuple

Frame = namedtuple("Frame", ["bytes", "timestamp", "duration"])


def read_wave(path):
    """
    Reads a 16-bit PCM mono 16kHz WAV file.

    :param path: str
        Path to the .wav file.

    :return: tuple
        (audio_bytes: bytes, sample_rate: int)
        Raw PCM audio data and its sample rate.
    """
    with wave.open(path, 'rb') as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            sys.exit("Error: wav file must be PCM16 mono 16kHz")
        frames = wf.readframes(wf.getnframes())
    return frames, wf.getframerate()


def frame_generator(frame_duration_ms, audio, sample_rate):
    """
    Splits raw audio into fixed-duration frames.

    :param frame_duration_ms: int
        Duration of each frame in milliseconds (e.g. 30).
    :param audio: bytes
        Raw PCM audio data.
    :param sample_rate: int
        Sampling rate of the audio (e.g. 16000).

    :return: generator
        Yields Frame namedtuples: (bytes, timestamp, duration)
    """
    n_bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * 2)  # 2 bytes/sample
    offset = 0
    timestamp = 0.0
    duration = frame_duration_ms / 1000.0
    while offset + n_bytes_per_frame <= len(audio):
        yield Frame(audio[offset:offset + n_bytes_per_frame], timestamp, duration)
        timestamp += duration
        offset += n_bytes_per_frame


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """
    Filters a stream of audio frames to detect voiced segments using a VAD.

    :param sample_rate: int
        Sample rate of the audio (e.g. 16000).
    :param frame_duration_ms: int
        Duration of each frame in milliseconds.
    :param padding_duration_ms: int
        Length of the padding window used to smooth VAD decisions.
    :param vad: webrtcvad.Vad
        An instance of the WebRTC VAD.
    :param frames: iterable
        Iterable of Frame namedtuples (from `frame_generator`).

    :return: generator
        Yields tuples (start_ms, end_ms) marking the start and end times of detected speech segments.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False
    speech_start = 0.0

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            # If more than half of the frames in the ring buffer are speech, trigger
            num_voiced = sum(1 for f, speech in ring_buffer if speech)
            if num_voiced > 0.5 * len(ring_buffer):
                triggered = True
                # speech starts at the timestamp of the first frame in the buffer
                speech_start = ring_buffer[0][0].timestamp
                ring_buffer.clear()
        else:
            # already in speech
            if not is_speech:
                ring_buffer.append((frame, is_speech))
                if len(ring_buffer) > num_padding_frames:
                    ring_buffer.pop(0)
                num_unvoiced = sum(1 for f, speech in ring_buffer if not speech)
                # if more than half unvoiced, end of segment
                if num_unvoiced > 0.5 * len(ring_buffer):
                    speech_end = frame.timestamp + frame.duration
                    yield int(speech_start * 1000), int(speech_end * 1000)
                    triggered = False
                    ring_buffer.clear()
            else:
                # still in speech, reset buffer
                ring_buffer.clear()

    # if we end while still triggered, close segment
    if triggered:
        speech_end = frame.timestamp + frame.duration
        yield int(speech_start * 1000), int(speech_end * 1000)


