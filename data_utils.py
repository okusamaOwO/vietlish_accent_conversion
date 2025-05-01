import time
import os
import random
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_wav_to_torch, load_filepaths_and_text, transform
from text import text_to_sequence, cleaned_text_to_sequence
#import h5py

"""Multi speaker version"""
import os
import random
import torch
import numpy as np
from text import text_to_sequence, cleaned_text_to_sequence
import commons  # for intersperse if add_blank is used

class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
    1) Loads audio, speaker_id (optional), text (optional)
    2) Converts text to sequences
    3) Computes spectrograms and optionally loads speaker embeddings or WavLM features
    """

    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = getattr(hparams, "text_cleaners", [])
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.cleaned_text = getattr(hparams, "cleaned_text", False)
        self.add_blank = getattr(hparams, "add_blank", False)
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.use_spk = getattr(hparams.model, "use_spk", False)
        self.spec_len = getattr(hparams.train, "max_speclen", None)

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """Filter samples by text length and store spectrogram lengths."""
        audiopaths_sid_text_new = []
        lengths = []
        for audiopath, sid, text in self.audiopaths_sid_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audiopaths_sid_text_new.append([audiopath, sid, text])
                lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def get_sid(self, sid):
        return torch.LongTensor([int(sid)]) if sid is not None else None

    def get_audio(self, filename):
        audio, sr = load_wav_to_torch(filename)
        if sr != self.sampling_rate:
            raise ValueError(f"Sample rate {sr} doesn't match target {self.sampling_rate}")

        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)

        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                                     self.sampling_rate, self.hop_length,
                                     self.win_length, center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)

        # Load WavLM feature
        c_filename = filename.replace(".wav", ".pt").replace("DUMMY", "dataset/wavlm")
        c_audio = torch.load(c_filename).squeeze(0)

        # Load speaker embedding
        if self.use_spk:
            spk_filename = filename.replace(".wav", ".npy").replace("DUMMY", "dataset/spk")
            spk = torch.from_numpy(np.load(spk_filename))
            return c_audio, spec, audio_norm, spk

        return c_audio, spec, audio_norm

    def __getitem__(self, index):
        entry = self.audiopaths_sid_text[index]
        audiopath, sid, text = entry if len(entry) == 3 else (entry[0], None, None)

        # Get optional components
        c_audio, spec, audio_norm, spk = self.get_audio(audiopath) if self.use_spk else (*self.get_audio(audiopath), None)
        c_text = self.get_text(text) if text is not None else None

        # Return tuple depending on presence of text/sid/spk
        out = [t for t in [c_text, c_audio, spec, audio_norm, spk] if t is not None]
        return tuple(out)

    def __len__(self):
        return len(self.audiopaths_sid_text)

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, hps):
        self.hps = hps
        self.use_spk = hps.model.use_spk

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, audio_normalized, spec_normalized, wav_normalized, spk]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        
        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[2].size(1) for x in batch])
        max_wav_len = max([x[3].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        if self.use_spk:
            spks = torch.FloatTensor(len(batch), batch[0][3].size(0))
        else:
            spks = None

        text_padded = torch.LongTensor(len(batch), max_text_len)
        c_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][2].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)

        text_padded.zero_()
        c_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
            
            c = row[1]
            c_padded[i, :, :c.size(1)] = c

            spec = row[2]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[3]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            if self.use_spk:
                spks[i] = row[4]
        
        spec_seglen = spec_lengths[-1] if spec_lengths[-1] < self.hps.train.max_speclen + 1 else self.hps.train.max_speclen + 1
        wav_seglen = spec_seglen * self.hps.data.hop_length 

        spec_padded, ids_slice = commons.rand_spec_segments(spec_padded, spec_lengths, spec_seglen)
        wav_padded = commons.slice_segments(wav_padded, ids_slice * self.hps.data.hop_length, wav_seglen)
        
        c_padded = commons.slice_segments(c_padded, ids_slice, spec_seglen)[:,:,:-1]
    
        spec_padded = spec_padded[:,:,:-1]
        wav_padded = wav_padded[:,:,:-self.hps.data.hop_length]

        if self.use_spk:
          return text_padded, c_padded, spec_padded, wav_padded, spks
        else:
          return text_padded, c_padded, spec_padded, wav_padded



class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
