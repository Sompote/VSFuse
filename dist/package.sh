#!/bin/bash
# Builds the release packages: macOS universal app (Intel + Apple silicon) and Windows x64.
# Needs the Rust targets aarch64-apple-darwin, x86_64-apple-darwin and x86_64-pc-windows-gnu,
# and mingw-w64 for the Windows link step. Run from the repository root: dist/package.sh
set -e
V=$(grep '^version' Cargo.toml | head -1 | cut -d'"' -f2)
OUT=dist/out; rm -rf "$OUT"; mkdir -p "$OUT"
for t in x86_64-apple-darwin aarch64-apple-darwin x86_64-pc-windows-gnu; do cargo build --release --target $t; done

# ---- macOS universal .app
APP="$OUT/VsFuse.app"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
lipo -create target/x86_64-apple-darwin/release/vsfuse target/aarch64-apple-darwin/release/vsfuse -output "$APP/Contents/MacOS/vsfuse"
ICON=$(mktemp -d)/VsFuse.iconset; mkdir -p "$ICON"
for s in 16 32 128 256 512; do
  sips -z $s $s assets/app_icon.png --out "$ICON/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) assets/app_icon.png --out "$ICON/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICON" -o "$APP/Contents/Resources/VsFuse.icns"
cat > "$APP/Contents/Info.plist" <<P
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>VsFuse</string>
<key>CFBundleDisplayName</key><string>VsFuse</string>
<key>CFBundleIdentifier</key><string>io.github.sompote.vsfuse</string>
<key>CFBundleExecutable</key><string>vsfuse</string>
<key>CFBundleIconFile</key><string>VsFuse</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$V</string>
<key>CFBundleVersion</key><string>$V</string>
<key>LSMinimumSystemVersion</key><string>10.13</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
P
codesign --force --deep -s - "$APP"
cp README.md LICENSE "$OUT/"; cp -R sample "$OUT/sample"
(cd "$OUT" && zip -qry "VsFuse-$V-macos-universal.zip" VsFuse.app README.md LICENSE sample)

# ---- Windows x64
W="$OUT/VsFuse-$V-windows-x64"; mkdir -p "$W"
cp target/x86_64-pc-windows-gnu/release/vsfuse.exe "$W/VsFuse.exe"; cp README.md LICENSE "$W/"; cp -R sample "$W/sample"
(cd "$OUT" && zip -qr "VsFuse-$V-windows-x64.zip" "VsFuse-$V-windows-x64")
ls -la "$OUT"/*.zip; lipo -info "$APP/Contents/MacOS/vsfuse"
