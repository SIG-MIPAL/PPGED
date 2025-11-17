python test_ed.py configs/CaptainCook4d/cook4d_b2_n2_gmm_dmap.yaml ckpt/CaptainCook4D_dmap/cook4d_b2_n2_gmm_dmap_final >> log/captaincook4d_online/test_ed.log

# python metric_vis_cook4d.py  --dirname cook4d_b2_n2_gmm_final_1013/ -ed --dirs CaptainCook4D >> log/captaincook4d_online/metric.log
# 在线的评测代码
python metric_vis_cook4d.py --dirs CaptainCook4D_dmap --dirname cook4d_b2_n2_gmm_dmap_final -ed --online
python metric_vis_cook4d.py --dirs CaptainCook4D_dmap --dirname cook4d_b2_n2_gmm_dmap_final -ed --online >> log/captaincook4d_online/metric.log