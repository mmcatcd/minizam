import numpy as np
from scipy.io import wavfile
from . import params
from matplotlib import pyplot as plt

class Peak:
    def __init__(self, segment_index, frequency_bin, magnitude):
        self.segment_index = segment_index
        self.frequency_bin = frequency_bin
        self.magnitude = magnitude

    def __repr__(self):
        return f"Point(segment_index={self.segment_index}, frequency_bin={self.frequency_bin})"


class Hash:
    def __init__(self, anchor_frequency, peak_frequency, d_time, anchor_time_offset, file_name):
        self.anchor_frequency = anchor_frequency
        self.peak_frequency = peak_frequency
        self.d_time = d_time
        self.anchor_time_offset = anchor_time_offset
        self.file_name = file_name

    @property
    def key(self):
        return (self.anchor_frequency, self.peak_frequency, self.d_time)
    
    @property
    def value(self):
        return {"file_name": self.file_name, "offset": self.anchor_time_offset}


class Engine:
    def __init__(self):
        self._database = {}
        self.file_names = []
        self.plot = False

    @staticmethod
    def _read_file_mono(file_path):
        sample_rate, y = wavfile.read(file_path)
        y = y.sum(axis=1) / 2 if y.ndim > 1 else y # Convert from stereo to mono
        return sample_rate, y

    def _create_spectrogram(self,
        y,
        sample_rate,
        fft_window_size=params.FFT_WINDOW_SIZE,
        fft_size=params.FFT_SIZE,
    ):
        window_size_samples = int(sample_rate * fft_window_size)

        spectrogram = []
        for offset in range(0, len(y), window_size_samples):
            yf = np.fft.fft(y[offset : offset + window_size_samples], fft_size*2).real
            spectrogram.append(abs(yf[:fft_size]))

        spectrogram = np.array(spectrogram)

        return spectrogram

    def _get_fingerprint(
        self,
        spectrogram,
        num_bands=params.NUMBER_OF_PEAK_BANDS,
        max_fft_bin=params.MAX_TARGET_FREQUENCY // ((params.SAMPLE_RATE / 2) / params.FFT_SIZE),
    ):
        bands = [int(b) for b in np.logspace(0, np.log2(max_fft_bin), num_bands, base=2)]

        peaks = []
        for segment_index, segment in enumerate(10*np.log10(spectrogram)):
            peak_candidates = [
                Peak(
                    segment_index=segment_index,
                    frequency_bin=start_band + np.argmax(segment[start_band:end_band]),
                    magnitude=np.max(segment[start_band:end_band]),
                ) for start_band, end_band in zip(bands[:-1], bands[1:])
            ]
            average_peak_magnitude = np.mean([peak.magnitude for peak in peak_candidates])
            peaks.extend([peak for peak in peak_candidates if peak.magnitude > average_peak_magnitude])

        if self.plot:
            xs, ys = zip(*[(p.segment_index, p.frequency_bin) for p in peaks])
            plt.figure(figsize=(15, 5))
            plt.scatter(xs, ys, s=1)
            plt.xlim(0, 200)
        
        return peaks

    @staticmethod
    def _get_hashes(peaks, file_name, region_offset=params.TARGET_REGION_OFFSET, region_size=params.TARGET_REGION_SIZE):
        hashes = []
        for idx, anchor_point in enumerate(peaks):
            if (idx + region_offset + region_size) < len(peaks):
                region_start = idx + region_offset
                region_end = region_start + region_size

                for region_peak in peaks[region_start:region_end]:
                    hashes.append(Hash(
                        anchor_frequency=anchor_point.frequency_bin,
                        peak_frequency=region_peak.frequency_bin,
                        d_time=region_peak.segment_index-anchor_point.segment_index,
                        anchor_time_offset=anchor_point.segment_index,
                        file_name=file_name,
                    ))
        return hashes

    def _add_hashes_to_database(self, hashes):
        for h in hashes:
            self._database.setdefault(h.key, []).append(h.value)

    def _process(self, file):
        sample_rate, y = self._read_file_mono(file["path"])
        spectrogram = self._create_spectrogram(y, sample_rate)
        fingerprint = self._get_fingerprint(spectrogram)
        hashes = self._get_hashes(fingerprint, file["name"])
        return hashes

    def seed_database(self, files):
        for file in files:
            self.file_names.append(file["name"])
            hashes = self._process(file)
            self._add_hashes_to_database(hashes)

    def match(self, file_path):
        hashes = self._process({"path": file_path, "name": ""})

        matches = []
        for h in hashes:
            hash_matches = self._database.get(h.key, [])
            for hash_match in hash_matches:
                matches.append((h, hash_match))

        # Finding the file with the largest number of matches.
        max_matches = 0
        max_file_name = ""
        for file_name in self.file_names:
            time_offsets = [
                match_value["offset"] - hash_.anchor_time_offset
                for hash_, match_value in matches if match_value["file_name"] == file_name
            ]
            time_offset_histogram, _ = np.histogram(time_offsets, len(set(time_offsets)))
            if max(time_offset_histogram) > max_matches:
                max_matches = max(time_offset_histogram)
                max_file_name = file_name

        return matches, max_matches, max_file_name
