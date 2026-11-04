#!/bin/sh
set -e

base_url="https://essentia.upf.edu/models"

wget -nc "${base_url}/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb"
wget -nc "${base_url}/tempo/tempocnn/deeptemp-k16-3.pb"
for mood in aggressive happy party relaxed sad; do
    wget -nc "${base_url}/classification-heads/mood_${mood}/mood_${mood}-discogs-effnet-1.pb"
    wget -nc "${base_url}/classification-heads/mood_${mood}/mood_${mood}-discogs-effnet-1.json"
done

sha1sum --check <<EOF
16fe9f9915041e2ef209be30893a9b4a00a8ad69  discogs-effnet-bs64-1.pb
8a52418fbf8220bffc8e88d7ad602c86a375a6fe  deeptemp-k16-3.pb
c349f9f3fc1200469c7cba24026aa8422f5a3130  mood_aggressive-discogs-effnet-1.json
e48ee4b58dde269d1136a59962f731d153c917da  mood_aggressive-discogs-effnet-1.pb
838871d772d9ef32f720c2dafd6fe532ea8095ce  mood_happy-discogs-effnet-1.json
77e2a44ede8abe9e90bcae3a879aeba1f0971552  mood_happy-discogs-effnet-1.pb
fc2df54b64d7811bffc79ba014fce9eb1195ea1b  mood_party-discogs-effnet-1.json
5c14ca88119cbf1d216265dd329b7599f76fb68c  mood_party-discogs-effnet-1.pb
3a0170e4756362609228cca686d646b97c8be889  mood_relaxed-discogs-effnet-1.json
2538580a017e6a94a7c5dadf46506f60a142b36b  mood_relaxed-discogs-effnet-1.pb
0334d870cc6518401c536bf906e6e0c5bf52a42e  mood_sad-discogs-effnet-1.json
8a813ccae5e7f83298a0432972642ab3ce9f30f7  mood_sad-discogs-effnet-1.pb
EOF
