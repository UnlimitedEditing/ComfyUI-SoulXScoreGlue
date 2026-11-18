# ComfyUI-SoulXScoreGlue

Glue for [SoulX-Singer](https://github.com/Soul-AILab/SoulX-Singer) in ComfyUI
([Saganaki22/ComfyUI-SoulX-Singer](https://github.com/Saganaki22/ComfyUI-SoulX-Singer)): write a song line as plain text
(lyrics + MIDI notes + beats) and get the metadata the **SoulX-Singer Advanced** node needs, with exact timing on the song
timeline.

- **SoulX Score Target** - `word:midi:beats` text + bpm + start time -> target metadata JSON path.
  `_` is a rest, `-` holds the previous word on a new note. No `/` characters, so it passes Graydient's option parser.
- **SoulX Prompt Voice** - the SoulX-Singer example English reference voice (audio + metadata), so no preprocessing
  models (speech recognition, separation) are needed.

Apache-2.0 (the bundled voice sample and the metadata format come from SoulX-Singer, Apache-2.0).
