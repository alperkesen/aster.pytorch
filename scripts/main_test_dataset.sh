CUDA_VISIBLE_DEVICES=0 python demo.py \
  --height 64 \
  --width 256 \
  --voc_type ALLCASES_SYMBOLS \
  --arch ResNet_ASTER \
  --with_lstm \
  --max_len 100 \
  --STN_ON \
  --beam_width 5 \
  --tps_inputsize 32 64 \
  --tps_outputsize 32 100 \
  --tps_margins 0.05 0.05 \
  --stn_activation none \
  --num_control_points 20 \
  --resume ../demo.pth.tar \
  --test_path ../data.jpg
