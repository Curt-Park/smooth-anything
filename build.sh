#!/bin/sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/chiaki-ng"
[ -f "$SRC/CMakeLists.txt" ] || git -C "$HERE" submodule update --init --recursive
cd "$SRC"
# An older version of the patch may be applied: restore the files it touches, then apply the current one.
if ! git apply --reverse --check "$HERE/smooth-hook.patch" 2>/dev/null; then
  git checkout -- $(git apply --numstat "$HERE/smooth-hook.patch" | cut -f3)
  git apply "$HERE/smooth-hook.patch"
fi
[ -d build-venv ] || { python3 -m venv build-venv && ./build-venv/bin/pip install -q "protobuf>=5,<6" setuptools; }
P="$(brew --prefix)"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHIAKI_ENABLE_CLI=OFF -DCHIAKI_ENABLE_STEAMDECK_NATIVE=OFF \
  -DPYTHON_EXECUTABLE="$PWD/build-venv/bin/python" -DSMOOTH_ANYTHING_DIR="$HERE" \
  -DOPENSSL_ROOT_DIR="$P/opt/openssl@3" -DOPENSSL_INCLUDE_DIR="$P/opt/openssl@3/include" \
  -DOPENSSL_CRYPTO_LIBRARY="$P/opt/openssl@3/lib/libcrypto.dylib" -DOPENSSL_SSL_LIBRARY="$P/opt/openssl@3/lib/libssl.dylib" \
  -DCMAKE_PREFIX_PATH="$P/opt/openssl@3;$P/opt/qt@6;$P/opt/protobuf@29"
if [ -L "$P/include/openssl" ] && readlink "$P/include/openssl" | grep -q "openssl@4"; then
  brew unlink openssl@4 >/dev/null
  trap 'brew link openssl@4 >/dev/null' EXIT
fi
CPATH="$P/opt/ffmpeg/include" PATH="$PWD/build-venv/bin:$PATH" cmake --build build --config Release --target chiaki
cd "$HERE"
clang -fobjc-arc -O2 -Wall -mmacosx-version-min=26.0 verify.m -o verify \
  -framework Cocoa -framework ScreenCaptureKit -framework CoreMedia -framework CoreVideo
echo "built $SRC/build/gui/chiaki.app and ./verify"
