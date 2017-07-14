#!/bin/bash

mkdir -p ../results/sintel

avg_epe="0"
avg_epe_occ="0"

for (( i=1; i <= 7; i++ )); do
    echo -n "Computing disparity between ${i}-imgL.ppm and ${i}-imgR.ppm... "
    ./sgm_gpu ../data/sintel/${i}-imgL.ppm ../data/sintel/${i}-imgR.ppm ../results/sintel/${i}-disparity.float3 >/dev/null 2>&1

    epe[$i]="$(./disp-epe ../results/sintel/${i}-disparity.float3 ../data/sintel/${i}-gt.float3)"
    echo -n "epe = ${epe[$i]} "
    avg_epe=$(echo "$avg_epe + ${epe[$i]}" | bc)

    epe_occ[$i]="$(./disp-epe ../results/sintel/${i}-disparity.float3 ../data/sintel/${i}-gt.float3 ../data/sintel/${i}-occ.pgm)"
    echo "epe (no occlusions) = ${epe_occ[$i]}"
    avg_epe_occ=$(echo "$avg_epe_occ + ${epe_occ[$i]}" | bc)
done

echo "Average EPE                : $(echo "scale=4; ${avg_epe} / 7.0" | bc)"
echo "Average EPE (no occlusions): $(echo "scale=4; ${avg_epe_occ} / 7.0" | bc)"

exit 0
