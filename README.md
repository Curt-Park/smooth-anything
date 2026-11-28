# remoteplay-60fps

30 → 60 fps frame interpolation for PS5 Remote Play on macOS.

Many PS5 games run at 30 fps, and some (GTA 6) have no 60 fps mode at all. remoteplay-60fps is built
into [chiaki-ng](https://github.com/chiaki-ng/chiaki-ng), the open-source Remote Play client, and
inserts a generated frame between every two game frames, so the game plays at 60 fps on a Mac. It
uses only Apple frameworks (VideoToolbox, Metal, Core Animation).

MIT licensed, except `smooth-hook.patch`, which modifies chiaki-ng and is under chiaki-ng's
AGPL-3.0 (with its OpenSSL exception). A binary built with chiaki-ng is covered by the AGPL-3.0 as
well. Not affiliated with Sony.

## Requirements

- Apple silicon Mac, macOS 27. 1080p interpolation needs an M2 Pro or faster; an M1 uses 720p.
- A PS5 registered in the app, streaming 1080p SDR
- Local Network access, or the app cannot reach the PS5. The downloaded app asks on first launch;
  a build run from a terminal uses the terminal's access (System Settings > Privacy & Security >
  Local Network)
- To build: Xcode command line tools and Homebrew

## Build and run

Download the DMG from [Releases](https://github.com/Curt-Park/remoteplay-60fps/releases) and drag the
app to Applications. It is not notarized, so on first launch allow it under System Settings >
Privacy & Security > Open Anyway. To build it yourself:

```sh
brew install qt@6 ffmpeg pkgconf opus openssl@3 cmake ninja nasm protobuf@29 speexdsp libplacebo \
             json-c miniupnpc libevent sdl2 vulkan-loader molten-vk
git submodule update --init --recursive
./build.sh
./run.sh
```

`build.sh` applies `smooth-hook.patch` to the chiaki-ng submodule (v1.10.1), replacing an older
version of the patch if one is applied, and builds it.
`run.sh` starts it with the Vulkan loader set and passes arguments on. To check what really reaches
the screen, run `./verify --match pid:<chiaki pid>` (needs Screen Recording permission).
`./package.sh <version>` builds the self-contained app, its DMG and a source archive with chiaki-ng
in `dist/`.

## How it works

1. **Take** each frame chiaki-ng decodes (`SmoothSubmit` in `smooth.m`) instead of its own view.
2. **Drop repeats.** A 30 fps game in a 60 fps stream sends every frame twice; the second copy is
   recognised by its small difference from the first.
3. **Interpolate** the midpoint of two game frames with `VTLowLatencyFrameInterpolation` at 1080p.
   If a warm-up call takes over 25 ms, it uses 720p and upscales the result with
   `VTLowLatencySuperResolutionScaler`. Scene cuts get no middle frame, and neither does a frame that
   arrives too soon after the previous one for a refresh to fit between them (menus, near-still
   pictures whose copies all read as new); otherwise such frames booked two refreshes each and the
   delay grew past 200 ms.
4. **Schedule** the middle frame and the untouched real frame on consecutive display refreshes.
5. **Present** through a Metal layer in a child window over chiaki-ng, driven by a display link and
   synced to the display refresh. Unsynced, a full-screen picture changed mid-scan and tore along a
   fixed line during camera moves. Every frame, real or generated, is upscaled to the window's pixels
   with MetalFX spatial scaling and then given contrast-adaptive sharpening; Core Animation's
   bilinear stretch of the 1080p frame looked soft. The upscale stops at 2x: on a 3008x1692
   ("looks like") 4K monitor the full 6016-pixel backing took 10–14 ms of GPU time a frame next to the
   interpolator, against about 5 ms at 2x, and looked the same.

The added delay is about 55–70 ms after a frame is decoded.

## Environment variables

Set them before `./run.sh`, or before `/Applications/remoteplay-60fps.app/Contents/MacOS/chiaki` for the
downloaded app.

| Variable | Effect |
|---|---|
| `SMOOTH_OFF=1` | Run as plain chiaki-ng |
| `SMOOTH_INTERP_720=1` | Interpolate at 720p even on a Mac fast enough for 1080p |
| `SMOOTH_FLOW=1` | Experimental: make middle frames by moving the stream's own pixels half way along `VTOpticalFlow` (run on 640x360 copies) instead of the interpolation network. Meant for Macs that fall back to 720p interpolation, whose network loses background texture in fast camera moves |
| `SMOOTH_UPSCALE=0` | Let Core Animation stretch the frame instead of MetalFX upscaling and sharpening |
| `SMOOTH_SHARPEN=<0–1>` | Sharpening strength after the MetalFX upscale. Default 0.7; 0 turns it off |
| `SMOOTH_DEBUG=1` | Print frame rates, timings and delay every two seconds, and how many refreshes after its target each frame reached the screen (a steady offset is compositor latency; `shifts` counts changes, which show as judder) |
| `SMOOTH_DIFF=1` | Print how each decoded frame was classified, with its arrival time and the values the decision used |
| `SMOOTH_DUMP=<dir>` | Save up to 15 raw NV12 triples (previous real frame, middle, next real frame) during fast motion, for offline comparison. When interpolation runs below the stream's size, the two real frames are also saved at the stream's size (`na`, `nb`) |
