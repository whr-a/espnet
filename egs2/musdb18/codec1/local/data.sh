#!/usr/bin/env bash

set -e
set -u
set -o pipefail

log() {
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

help_message=$(cat << EOF
Usage: $0 [--sample_rate <44.1k>] [--nchannels <1/2>]
  optional argument:
    [--use_hq]: whether to download (uncompressed) high-quality audios (Default: false)
        Note: Samples in musdb18hq and musdb18 have slightly different durations.
    [--sample_rate]: sampling rate of the data (Default: 44.1k)
    [--nchannels]: number of channels of the data (Default: 1)
EOF
)

. ./db.sh
. ./path.sh

use_hq=false
sample_rate=44.1k
nchannels=1
dev_names=(
    "Actions - One Minute Smile"
    "Clara Berry And Wooldog - Waltz For My Victims"
    "Johnny Lokke - Promises & Lies"
    "Patrick Talbot - A Reason To Leave"
    "Triviul - Angelsaint"
    "Alexander Ross - Goodbye Bolero"
    "Fergessen - Nos Palpitants"
    "Leaf - Summerghost"
    "Skelpolu - Human Mistakes"
    "Young Griffo - Pennies"
    "ANiMAL - Rockshow"
    "James May - On The Line"
    "Meaxic - Take A Step"
    "Traffic Experiment - Sirens"
)

stage=0
stop_stage=100

. utils/parse_options.sh

if [ $# -ne 0 ]; then
    echo "${help_message}"
    exit 1;
fi

if [ -z "${MUSDB18}" ]; then
    log "Fill the value of 'MUSDB18' of db.sh"
    exit 1
fi

if [ $nchannels -eq 1 ]; then
    log "Use single channel audios"
elif [ $nchannels -eq 2 ]; then
    log "Use stereo audios"
else
    log "Unsupported number of channels: ${nchannels}"
    exit 1
fi


cdir=$PWD


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Downloading MUSDB18 data to '${MUSDB18}'"

    mkdir -p "${MUSDB18}/"
    if ${use_hq}; then
        # uncompressed version (22.7GB unzip to 30GB)
        musdb18_url="https://zenodo.org/record/3338373/files/musdb18hq.zip?download=1"
        format=".wav"
        num_train=500
        num_test=250
    else
        # compressed version (4.7 GB unzip to 5.3GB)
        musdb18_url="https://zenodo.org/record/1117372/files/musdb18.zip?download=1"
        format=".mp4"
        num_train=100
        num_test=50
    fi
    if [ ! -e "${MUSDB18}/test" ] || [ ! -e "${MUSDB18}/train" ]; then
        num_test_audios=0
        num_train_audios=0
    else
        num_test_audios=$(find "${MUSDB18}/test" -iname "*${format}" | wc -l)
        num_train_audios=$(find "${MUSDB18}/train" -iname "*${format}" | wc -l)
    fi
    if [ "${num_test_audios}" = "${num_test}" ] && [ "${num_train_audios}" = "${num_train}" ]; then
        echo "'${MUSDB18}/' already exists. Skipping..."
    else
        wget --continue -O "${MUSDB18}/musdb18.zip" ${musdb18_url}
        unzip "${MUSDB18}/musdb18.zip" -d "${MUSDB18}"
    fi

    if ! ${use_hq}; then
        log "Converting mp4 to wav format..."

        if ! command -v ffmpeg &> /dev/null; then
            log "Please install ffmpeg first"
            exit 1
        fi
        # convert multi-stream mp4 to multiple wav audios
        for dset in train test; do
            find "${MUSDB18}/${dset}" -iname "*${format}" | while read -r fname; do
                fbasename=$(basename "${fname}" .stem.mp4)
                outdir="${cdir}/data/musdb18/${dset}/${fbasename}"
                if [ ! -d "$outdir" ]; then
                    mkdir -p "${outdir}"
                    if [ $nchannels -eq 1 ]; then
                        ffmpeg -nostdin -hide_banner -loglevel warning -i "${fname}" \
                            -map 0:a:0 -ac 1 -channel_layout mono \
                            -c pcm_s16le "${outdir}/mixture.wav"
                    else
                        ffmpeg -nostdin -hide_banner -loglevel warning -i "${fname}" \
                            -map 0:a:0 -c pcm_s16le "${outdir}/mixture.wav"
                    fi
                fi
            done
        done
    else
        mkdir -p "${cdir}/data"
        ln -s "$MUSDB18" "${cdir}/data/musdb18"
    fi
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Data preparation"

    for x in train test; do
        dset="${x}_${sample_rate}_${nchannels}ch"
        mkdir -p "${cdir}/data/${dset}"
        find "${cdir}/data/musdb18/${x}" -iname "mixture.wav" > "${cdir}/data/${dset}/flist"
        awk -F"/" '{print(tolower($(NF-1)))}' "${cdir}/data/${dset}/flist" | sed -e 's/ /_/g' > "${cdir}/data/${dset}/uid"
        paste -d' ' "${cdir}/data/${dset}/uid" "${cdir}/data/${dset}/flist" | sort -u > "${cdir}/data/${dset}/wav.scp"
        awk '{print($1 " " $1)}' "${cdir}/data/${dset}/wav.scp" > "${cdir}/data/${dset}/utt2spk"
        utils/utt2spk_to_spk2utt.pl "${cdir}/data/${dset}/utt2spk" > "${cdir}/data/${dset}/spk2utt"

        rm "${cdir}/data/${dset}/flist" "${cdir}/data/${dset}/uid"
    done

    mv "${cdir}/data/train_${sample_rate}_${nchannels}ch" "${cdir}/data/train_all"
    mkdir -p "${cdir}/data/train_${sample_rate}_${nchannels}ch" "${cdir}/data/dev_${sample_rate}_${nchannels}ch"
    grep -F -f <(printf '%s\n' "${dev_names[@]}") "${cdir}/data/train_all/wav.scp" > "${cdir}/data/dev_${sample_rate}_${nchannels}ch/wav.scp"
    grep -v -F -f <(printf '%s\n' "${dev_names[@]}") "${cdir}/data/train_all/wav.scp" > "${cdir}/data/train_${sample_rate}_${nchannels}ch/wav.scp"
    for x in train dev; do
        utils/filter_scp.pl ${cdir}/data/${x}_${sample_rate}_${nchannels}ch/wav.scp <${cdir}/data/train_all/utt2spk > ${cdir}/data/${x}_${sample_rate}_${nchannels}ch/utt2spk
        utils/filter_scp.pl ${cdir}/data/${x}_${sample_rate}_${nchannels}ch/utt2spk <${cdir}/data/train_all/wav.scp > ${cdir}/data/${x}_${sample_rate}_${nchannels}ch/wav.scp
        utils/utt2spk_to_spk2utt.pl "${cdir}/data/${x}_${sample_rate}_${nchannels}ch/utt2spk" > "${cdir}/data/${x}_${sample_rate}_${nchannels}ch/spk2utt"
    done

    for x in train dev test; do
        utils/validate_data_dir.sh --no-text --no-feats "${cdir}/data/${x}_${sample_rate}_${nchannels}ch"
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
