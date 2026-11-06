#!/bin/bash
# Rebuilds data/grabber: images tagged the way Grabber writes them
# (exiftool -charset filename=utf8 -sep ";" -KEY=VALUE -m -overwrite_original). Needs exiftool, Python with Pillow,
# and a short H.264 clip as $2 (e.g. clip_h264.mp4 from: python tests/make_media.py <dir>).
set -e
D=$1; CLIP=$2; rm -rf "$D" && mkdir -p "$D" && cd "$D"
python -c "
from PIL import Image
for i, n in enumerate(['a01.jpg','a02.jpg','a03.jpg','a04.png','a05.png','a07.jpg','a08.webp','a09.jpg','a10.jpg','a11.jpg','a12.jpg','a13.png','a15.jpg']):
    Image.new('RGB', (900, 600), (i * 15, 80, 120)).save(n)
Image.new('RGB', (600, 900), (50, 50, 50)).save('a06.gif')"
cp "$CLIP" a14.mp4
g() { exiftool -charset filename=utf8 -sep ";" "$@" -m -overwrite_original >/dev/null 2>&1; }
g "-Keywords=1girl,long_hair,hatsune_miku,vocaloid,smile" a01.jpg                                   # comma style -> IPTC
g "-Keywords=1girl,long_hair,smile" "-Subject=1girl,long_hair,smile" a02.jpg                         # both fields
g "-Subject=artist:kantoku;character:hatsune_miku;copyright:vocaloid;general:long_hair;general:1girl;general:smile;meta:highres" a03.jpg  # %all:includenamespace,unsafe,separator=;%
g "-Subject=1girl long_hair hatsune_miku artist_name" a04.png                                       # default %all%
g "-Subject=artist_kantoku character_hatsune_miku copyright_vocaloid general_1girl general_artist_name" a05.png   # %all:includenamespace%
g "-Keywords=1girl,solo,animated" a06.gif                                                           # GIF Keywords -> XMP-pdf
g "-Subject=1girl,初音ミク,smile" a07.jpg                                                            # non-ASCII via XMP
exiftool -charset filename=utf8 -sep ";" "-Subject=1girl,webp_sidecar" -o a08.webp.xmp >/dev/null 2>&1   # sidecar image.ext.xmp
exiftool -charset filename=utf8 -sep ";" "-Subject=1girl,noext_sidecar" -o a09.xmp >/dev/null 2>&1       # sidecar image.xmp
g "-XPKeywords=1girl;smile;score_9" a10.jpg                                                          # Windows keywords
:                                                                                                    # a11.jpg: no metadata
g "-Subject=^_^,@_@,1girl" a12.jpg                                                                   # kaomoji
g "-Keywords=yuri,2girls,multiple_girls" a13.png                                                     # IPTC inside PNG
g "-Subject=1girl,video_frame" a14.mp4                                                               # video source
g "-Subject=hatsune_miku_(cosplay),1girl" a15.jpg                                                    # parentheses
