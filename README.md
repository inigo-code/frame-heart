# Frame Heart

![The Frame Heart window, with a before and after viewer](https://inigo-code.github.io/frame-heart-site/assets/remove-wide.webp)

Version **1.2.3**.

A portable Windows app that finds and removes a visible corner watermark on AI images and videos, and can enlarge images with Real-ESRGAN. One of the [Heart Apps](https://inigo-code.github.io/).

It handles the Gemini sparkle, and the corner labels from Grok, Doubao, Jimeng, Tongyi, Qwen, Kling, Wenxin, and Veo. It works particularly well on images and videos made with Gemini. Detection is automatic. The Gemini mark is reversed, so the pixels underneath are restored. A label is rebuilt from the pixels around it. It does not touch SynthID or other invisible provenance marks.

Original files are never overwritten. New copies go to a separate folder.

Use it on pictures and videos you made or have the right to edit.

## Start

Double-click `Start.bat`.

The first launch sets up a local `.venv` in this folder. That needs Python 3.11 or newer on the machine. Later launches start the window directly.

If `portable\FrameHeart\FrameHeart.exe` is present, `Start.bat` opens that instead. The exe does not need Python. The libraries that used to sit in `_internal` are packed inside `FrameHeart.exe`. Keep the `tools` folder beside the exe for video, and the `models` folder for models you download. Windows 10 or 11 with the WebView2 runtime is required. The first open unpacks into a temporary folder, so the window takes longer to appear.

## Window

The sidebar has Clean, Upscale, Queue, and Settings.

1. Choose a folder, use **Add files**, or drop images, videos, or a folder onto the window.
2. Confirm **Save to**. It defaults to a sibling folder named `YourFolder_cleaned`, or `YourFolder_upscaled` when pictures are being enlarged.
3. On Clean, press **Clean**. Frame Heart finds the mark by itself. Turn on **Upscale after cleaning** to enlarge pictures in the same pass. Videos are cleaned, not enlarged.
4. On Upscale, pick a model. **Download** fetches one that is not included. **Add your own** takes an ncnn `.param` and `.bin` pair.
5. **Preview** checks the first file. Drag across the picture to compare before and after.
6. **Queue** lists every file. Pressing Stop halts the batch, and starting again carries on from the file it stopped on. **Cancel** stops the file that is running, and **Take off list** drops a file from the batch. **Clear finished** drops files that already ran, and **Clear list** empties it. None of these touch your files.

Settings holds the processor, JPEG and WebP quality, video compression, and the audio track. Upscale uses the graphics processor when GPU is selected and Vulkan is available. CPU is slower. A download keeps the rest of the window usable.

## Video

Clean looks for the mark on up to seven frames across the clip. When it finds one, it studies frames from the whole clip to learn the star this clip actually carries, so dark scenes come out clean too. A frame that does not carry the mark is left as it was. A clip with no mark is copied unchanged. The soundtrack is kept unless **Keep the audio track** is turned off. The clip's color tags carry over to the new file.

Videos are not enlarged. **Upscale** skips a video. **Clean and upscale** cleans it and keeps its size. Pictures in the same batch are still enlarged.

Output is H.264. MP4, MOV, and MKV keep that container. WebM, M4V, and other files are saved as MP4. Kept audio is AAC at 192 kbps. Compression defaults to 18. When the frame fits, the NVIDIA encoder writes it. Larger frames, and computers without that encoder, use libx264 at the veryfast preset.

This needs FFmpeg. If it is not already on the computer, Settings offers **Download FFmpeg**. The tools are saved in `FrameHeart\tools`, next to the exe. If the exe is already inside a folder named FrameHeart, `tools` sits directly in that folder.

## Models

The model list is on Upscale.

- **Photo** and **Anime** are included. Photo is Real-ESRGAN x4plus. Anime is the lighter anime model, at 2× and 4×. Photo at 2× runs the 4× model and fits the result down to 2×. The same fit is available for every 4× model.
- **General**, **Compact**, **Anime detail**, **Nomos8k**, **Siax**, **UltraSharp**, **AnimeSharp**, and **Remacri** show a size. Choose one and click **Download**. The files are saved in `FrameHeart\models` and stay there for next time. If the exe is already inside a folder named FrameHeart, the models go in that folder's `models` directory.
- UltraSharp and Remacri come from the Upscayl model folder. Remacri is the 4× Foolhardy Remacri model. AnimeSharp comes from the realesrgan-gui additional-models release.
- **Add model** takes your own ncnn `.param` and `.bin` pair. The file name has to contain x2, x3, or x4.

General, Compact, and Anime are the faster models. Photo, Anime detail, Nomos8k, Siax, UltraSharp, AnimeSharp, and Remacri are slow on CPU.

JPEG and WebP files are saved again at the quality set under Pictures. PNG stays lossless. For the sharpest result, start from PNG. Video compression defaults to 18. A higher number makes a smaller file.

## Command line

```
python app.py INPUT OUTPUT --recursive --quality 95
python app.py INPUT OUTPUT --task upscale --model photo --scale 4 --device gpu
python app.py INPUT OUTPUT --combine --model anime --scale 2
python app.py INPUT OUTPUT --task clean --video-crf 18
```

`INPUT` can be one image, one video, or a folder. `--task` is `clean` (the default) or `upscale`. `--combine` removes the watermark and upscales a picture in one pass. A video is cleaned and not enlarged. `--task upscale` skips videos. `--mark` picks one platform and `--mode` picks a fixed Gemini corner, `standard` or `current`. Leave both out: Auto finds the mark, and a fixed corner is only used when Auto finds nothing. `--model` is `photo`, `anime`, `general`, `anime-detail`, `compact`, `nomos`, `siax`, `ultrasharp`, `animesharp`, `remacri`, or the id of a model you added. `--scale` is `2`, `3`, or `4` when that model supports it. Gemini layout modes are `auto`, `standard`, and `current`. `--device` is `gpu` or `cpu`. `--video-crf` is 16 to 28. `--no-audio` leaves the soundtrack out. A model that is not downloaded yet has to be downloaded from the window first. FFmpeg has to be on the PATH or in the `tools` folder.

## Limits

- Visible marks only. SynthID is untouched. ChatGPT, Midjourney, and Firefly do not stamp a corner logo this tool can reverse.
- A label has to be see-through and sit close to a corner, the way generators place them. Solid text is treated as part of the picture, so a signature or a caption stays. A label on a detailed background is rebuilt from its surroundings, so a close look can still find the spot.
- Text in a video has to stay in one place to count as a mark.
- A star that has been heavily cropped, or sits outside the corner, is not found.
- Photo, Anime detail, Nomos8k, Siax, UltraSharp, AnimeSharp, and Remacri are slow without a Vulkan GPU. General, Compact, and Anime are faster. Choose CPU in Settings, or pass `--device cpu`, to upscale on the CPU.
- General, Compact, Nomos8k, and Siax download from the Upscayl custom-models repository. UltraSharp and Remacri download from the Upscayl application model folder. AnimeSharp downloads from the realesrgan-gui additional-models release. Anime detail downloads the official Real-ESRGAN ncnn package and keeps only that model. The first download needs a network connection.

## Source

This code is published so you can read it and build it yourself. All rights reserved: there is no open-source licence. Third-party parts keep their own licences, listed in [NOTICE](NOTICE).

The bundled Photo and Anime models in `assets/models` aren't in this repository because of their size. Add them from the Real-ESRGAN ncnn release before building, or the app will offer only the models it can download.

Downloads and the website: [Frame Heart](https://inigo-code.github.io/frame-heart-site/). Bugs and ideas: inigo@tosstheworld.org.
