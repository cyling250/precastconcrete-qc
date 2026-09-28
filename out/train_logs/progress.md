2026-09-17 07:00 | step01 phase1_pointmlp (RUNNING) | epoch 61/80, cm=0.9265 (unified caliber, tracking manuscript 0.9237) | queue healthy, no FAIL, ~6.3 min/epoch, ETA step1 ~09:00
2026-09-17 13:05 | step03 a2_m3_single_stage (RUNNING) | epoch 6/120, cm=0.8298 | step1 OK (best mIoU 0.903@ep72), step2 OK: test mIoU/cm/rsIoU = 0.9183/0.9231/0.9269 (manuscript 0.918/0.922/0.927 - near-exact reproduction). pointmlp weights READY -> E2/E5-deep unlocked for main session. No FAIL.
2026-09-17 19:05 | step03 a2_m3_single_stage (RUNNING) | process alive (CPU 30741s) | no FAIL, no restart needed
2026-09-17 19:05 | step03 a2_m3_single_stage (RUNNING) | epoch 53/120, cm=0.9159 | process alive, no FAIL
2026-09-18 01:05 | step03 a2_m3_single_stage (RUNNING) | epoch 100/120, cm=0.9283 | process alive, no FAIL
2026-09-18 07:05 | step04 a1a_m1_random_sampling (RUNNING) | epoch 37/120, cm=0.8161 | step03 M3 DONE: test mIoU/cm = 0.8998/0.9265 (manuscript 0.899/0.924); process alive, no FAIL
2026-09-18 (pause) | QUEUE PAUSED by user request for ZCode software update | paused at step04 a1a_m1 epoch ~72/120 (checkpoint intact) | resume: relaunch train_queue.py (auto-skips done steps, auto-resumes M1); 6h health-check automation stays active
2026-09-18 (health check) | QUEUE INTENTIONALLY PAUSED (user ZCode update) - no restart performed, awaiting user resume signal | state: 3/13 done, step04 a1a_m1 checkpoint 72/120 intact
2026-09-18 (handover pause) | QUEUE STOPPED CLEANLY for AI handover | 3/13 done, step04 a1a_m1 checkpoint 72/120 intact | next AI: relaunch train_queue.py to continue
2026-09-18 14:02 | (post-queue) | 训练队列未完成，post-queue 不启动（等下次触发）
2026-09-18 14:05 | step04 a1a_m1_random_sampling (RUNNING) | 13:34 的启动在 13:37 被外部杀死(无 traceback);13:53 已用 Start-Process 脱离会话重启,从 ep71 断点续训 | 每2小时健康检查自动化已重建(id automation-fd6a8db8);post_queue_runner.py 就绪,队列完成后自动接力 E4→E3→E1→E2(新旧口径)→E5-deep
2026-09-18 14:10 | step04 a1a_m1_random_sampling (RUNNING) | 应用户要求把队列从 Start-Process 脱离进程改为 ZCode 后台终端任务(run_in_background, id exec_d585c74c);旧进程树已 taskkill /T 干净杀掉,14:07 重启后从 ep71 断点续训 | 健康检查自动化的重启方式已同步改为后台任务方式
2026-09-18 14:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 72/120, mIoU=0.786, cm=0.831, 早停计数 2/10 (最佳 mIoU 0.788@ep70) | 进程存活,无需重启
2026-09-18 16:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 90/120, mIoU=0.791, cm=0.838, 早停计数 7/10 (最佳 mIoU 0.793@ep83) | 进程存活,无需重启
2026-09-18 18:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 92/120, mIoU=0.794, cm=0.839, 刷新最佳@ep92 (早停计数清零) | 进程存活,无需重启
2026-09-18 20:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 94/120, mIoU=0.792, cm=0.839, 早停计数 2/10 (最佳 mIoU 0.794@ep92) | 进程存活,无需重启
2026-09-18 22:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 97/120, mIoU=0.793, cm=0.841, 刷新最佳 mIoU 0.794@ep96 | 进程存活,无需重启
2026-09-19 00:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 100/120, mIoU=0.795, cm=0.841, 刷新最佳@ep100 (早停清零) | 进程存活,无需重启
2026-09-19 02:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 103/120, mIoU=0.793, cm=0.841, 早停计数 2/10 (最佳 mIoU 0.795@ep101) | 进程存活,无需重启
2026-09-19 04:23 | step04 a1a_m1_random_sampling (RUNNING) | epoch 106/120, mIoU=0.794, cm=0.841, 早停计数 5/10 (最佳 mIoU 0.795@ep101) | 进程存活,无需重启
2026-09-19 06:24 | step04 a1a_m1_random_sampling (RUNNING) | epoch 109/120, mIoU=0.794, cm=0.841, 早停计数 8/10 (最佳 0.795@ep101) | 进程存活,无需重启;注意:GPU 显存 15.5/16GB、约16个计算进程共享,训练速度降至约40min/epoch(此前6.7min)
2026-09-19 08:23 | step04 a1a_m1 OK→step05 a1b_m2 (RUNNING) | step04 完成: test mIoU=0.800/cm=0.848 (最佳val 0.795@ep101);step05 已到 ep6, 最佳 val mIoU 0.847@ep6 | 进程存活,无需重启;队列 4/13 步完成
2026-09-19 10:23 | step05 a1b_m2_no_pos_balance (RUNNING) | epoch 22/120, mIoU=0.881, cm=0.898, 早停计数 2/10 (最佳 0.884@ep20) | 进程存活,无需重启
2026-09-19 12:23 | step05 a1b_m2_no_pos_balance (RUNNING) | epoch 39/120, mIoU=0.893@ep37 刷新最佳, cm=0.915 | 进程存活,无需重启
2026-09-19 14:23 | step05 a1b_m2_no_pos_balance (RUNNING) | epoch 55/120, mIoU=0.900@ep53 最佳, cm=0.925, 早停计数 2/10 | 进程存活,无需重启
2026-09-19 16:23 | step05 a1b_m2_no_pos_balance (RUNNING) | epoch 72/120, mIoU=0.902, cm=0.928, 早停计数 4/10 (最佳 0.903@ep67) | 进程存活,无需重启
2026-09-19 18:23 | step05 a1b_m2 OK→step06 phase1_pointnet++ (RUNNING) | step05 完成: test mIoU=0.899/cm=0.927;step06 已到 ep6/80, val mIoU 0.761 上升中 | 进程存活,无需重启;队列 5/13 步完成
2026-09-19 20:23 | step06 phase1_pointnet++ (RUNNING) | epoch 19/80, mIoU=0.834, cm=0.838, 早停计数 3/10 (最佳 0.836@ep16) | 进程存活,无需重启
2026-09-19 22:23 | step06 phase1_pointnet++ (RUNNING) | epoch 32/80, mIoU=0.850, cm=0.864, 早停计数 2/10 (最佳 0.854@ep30) | 进程存活,无需重启
2026-09-20 00:23 | step06 phase1_pointnet++ (RUNNING) | epoch 44/80, mIoU=0.865@ep42 最佳, cm=0.887, 早停计数 2/10 | 进程存活,无需重启
2026-09-20 02:23 | step06 phase1_pointnet++ (RUNNING) | epoch 57/80, mIoU=0.870@ep55 最佳, cm=0.896, 早停计数 2/10 | 进程存活,无需重启
2026-09-20 04:23 | step06 phase1_pointnet++ (RUNNING) | epoch 70/80, mIoU=0.872, cm=0.899, 早停计数 6/10 (最佳 0.873@ep64) | 进程存活,无需重启
2026-09-20 06:23 | step06 phase1_pointnet++ OK→step07 phase2_pointnet++ (RUNNING) | step06 完成: test mIoU=0.869/cm=0.898 (S1 口径);step07 全管线已开跑,S2 val mIoU≈0.88 | 进程存活,无需重启;队列 6/13 步完成
2026-09-20 08:24 | step07 phase2_pointnet++ OK→step08 phase1_pointnext (RUNNING) | step07 完成: test mIoU=0.897/cm=0.903/rsIoU=0.912 (pointnet++ 全管线);step08 pointnext S1 刚启动 | 进程存活,无需重启;队列 7/13 步完成
2026-09-20 10:25 | ⚠ step07 phase1_pointnet++ FAIL(CUDA OOM,外部程序占显存)→step08 phase2_pointnext 连锁 FAIL(缺S1权重);step09 pointnet S1 OK(test mIoU=0.643);step10 phase2_pointnet RUNNING(~ep35) | 队列按'单步FAIL不阻塞'继续;计划:post_queue_runner 已加前置步,队列完成后自动重跑 train_queue.py 仅重试失败步
2026-09-20 10:25 | (post-queue) | 训练队列未完成，post-queue 不启动（等下次触发）
2026-09-20 12:24 | step10 phase2_pointnet OK→step12 a4_m4_focal (RUNNING) | step10 完成: test mIoU=0.718/cm=0.748/rsIoU=0.775 (pointnet 全管线);step12 focal 消融已启动 ep1 | 进程存活,无需重启;队列 9/13 步完成(另2步FAIL待队列后重试)
2026-09-20 14:23 | step12 a4_m4_focal (RUNNING) | epoch 11/120, mIoU=0.848, cm=0.870 连续刷新最佳 | 进程存活,无需重启
2026-09-20 16:23 | step12 a4_m4_focal (RUNNING) | epoch 26/120, mIoU=0.865, cm=0.886 (最佳 mIoU 0.869) | 进程存活,无需重启
2026-09-20 17:21 | step12 a4_m4_focal (RUNNING) | epoch 29/120, 最佳 mIoU 0.870, cm 最高 0.892 | 用户以为队列退出,核实:进程链健在(执行器26460/训练18732),GPU 100%/15.8GB,无需恢复;消失的是ZCode任务条目(会话重载后列表清空),进程本身脱离会话存活
2026-09-20 18:10 | step12 a4_m4_focal (已重启) | 应用户要求恢复后台可见:杀掉脱管进程链后以 run_in_background 重启(id exec_1cdbd432),从 checkpoint 断点续训 | 损失最多1个epoch
2026-09-20 18:24 | step07 phase1_pointnext (RUNNING,重试) | 18:09 起 重跑此前 OOM 失败的 pointnext S1(无 epoch 日志,首个epoch未完);更正:上次为恢复可见性的重启使队列从头顶重巡,step07→08 失败步先重试,a4_m4_focal(已至ep30)排在其后续训 | 进程存活(19056),无需重启
2026-09-20 18:52 | ⚠ step07 phase1_pointnext 重试再次 OOM→根因确认:b16 下需求约18.7GB(ball_query 排序8GB+已占10.75GB),16GB 卡确定性装不下;step08 phase2_pointnext 假 OK(缺S1权重时 rc=0 退出,实际未训练,但未写 summary.json 故后续会重试) | 用户拍板 pointnext S1 以 b8 补跑:已改 _s1_trainer(加 batch_size 参数,默认16不变)/phase1脚本(加--batch-size)/队列step07(--batch-size 8 --val-batch-mult 1)/phase2脚本(缺权重改 rc=1);队列跑完 a4→a5 后自动化 REQUEUE 将自动用新参数重试 07/08
2026-09-20 19:03 | (handover) 队列已干净停止(a4_m4_focal 断点 best val mIoU 0.873 完好),定时任务已删;10/13 步完成,pointnext b8 补跑参数已就绪 | 下一个 AI:按 handover.md §1 一条命令恢复队列 + 重建每2小时健康检查自动化
2026-09-20 19:12 | (handover恢复) 队列已用 run_in_background 重启(id exec_9502c2a9,执行器 PID 9564),19:09 起自动先补跑 step07 phase1_pointnext(b8 已生效: Batch: 8,GPU 7.6/16GB);a4_m4_focal 排其后断点续训 | 每2小时健康检查自动化已重建(id automation-cd150cd7,下次触发 21:23);后队列接力逻辑不变
2026-09-20 19:24 | step07 phase1_pointnext (RUNNING, b8 补跑) | 队列启动 14 分钟,ep1 train(375/375)+val(504/504) 刚跑完,指标行待刷(日志 mtime 19:23:21,仍在写入;GPU 7.6GB) | 进程存活(执行器9564/训练34044),无需重启;注意:自动化首个触发在 19:23(创建当日先对齐整点23分),之后每2小时(21:23起)
2026-09-20 21:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 11/80, 最佳 mIoU 0.585@ep9 (cm 最高 0.512,金属类 IoU 0.302 偏低属早期), 早停 1/10 | 进程存活(执行器9564/训练34044,GPU 97%/7.6GB),无需重启;约11分钟/epoch,step07 预计明日上午完,队列整体还需 step08/09 重跑+a4 断点+a5,后天节奏
2026-09-20 23:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 20/80, 最佳 mIoU 0.663@ep20 连续刷新 (cm 0.597;金属类 0.302→0.421 回升中), 早停清零 | 进程存活(执行器9564/训练34044,GPU 7.8GB,日志 mtime 23:23:06),无需重启;节奏符合 ~11分钟/epoch 预期
2026-09-21 01:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 29/80, 最佳 mIoU 0.707@ep28 (cm 0.657;金属类 0.507 持续回升), 早停 1/10 | 进程存活(执行器9564/训练34044,GPU 7.8GB,日志 mtime 01:19:59),无需重启
2026-09-21 03:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 39/80, 最佳 mIoU 0.739@ep38 连续刷新 (cm 0.705;金属类 0.577), 早停清零 | 进程存活(执行器9564/训练34044,GPU 7.8GB,日志 mtime 03:22:38),无需重启;实际约12.7分钟/epoch,step07 ETA 今日中午
2026-09-21 05:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 48/80, 最佳 mIoU 0.754@ep47 (cm 0.722;金属类 0.601), 早停清零 | 进程存活(执行器9564/训练34044,GPU 7.9GB,日志 mtime 05:18:09),无需重启;ETA 维持今日中午
2026-09-21 07:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 57/80, 最佳 mIoU 0.773@ep57 刚刷新 (cm 0.746;金属类 0.636), 早停清零 | 进程存活(执行器9564/训练34044,GPU 7.9GB,日志 mtime 07:19:53),无需重启;剩23 epoch,ETA ~12:15
2026-09-21 09:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 67/80, 最佳 mIoU 0.777@ep65 (cm 0.750;金属类 0.641), 早停 1/10 | 进程存活(执行器9564/训练34044,GPU 7.8GB,日志 mtime 09:20:12),无需重启;剩13 epoch,ETA ~12:10
2026-09-21 11:23 | step07 phase1_pointnext (RUNNING, b8 补跑) | epoch 75/80, 最佳 mIoU 0.780@ep69 (cm 0.754;金属类 0.648), 早停 5/10(平台期,不影响收尾) | 进程存活(执行器9564/训练34044,GPU 7.9GB,日志 mtime 11:16:24),无需重启;剩5 epoch,step07 ETA ~12:20
2026-09-21 13:23 | step07 OK→step08 phase2_pointnext (RUNNING) | **step07 b8 补跑完成: 最佳 mIoU=0.780, 测试 mIoU=0.777/cm=0.758**;step08 S2 已到 ep2, mIoU=0.772/cm=0.751/rsIoU=0.701 连续刷新最佳(S2纯 0.811) | 进程存活(执行器9564/新训练进程14860),无需重启;注意:phase2_pointnext 显存 15.8/16GB 偏高但已稳定跑过2个epoch,下次巡检重点看是否 OOM
2026-09-21 15:23 | step08 phase2_pointnext (RUNNING) | epoch 14, mIoU=0.831/cm=0.824/rsIoU=0.816 刷新最佳(S2纯 0.909;套筒类 0.727 上升中) | 进程存活(执行器9564/训练14860,GPU 15.8GB 持平,无 OOM),无需重启;~11.5分钟/epoch,本轮 12:42 起,预计深夜至明晨完,后续 step09 重跑(分钟级)+a4 断点+a5
2026-09-21 17:23 | step08 phase2_pointnext (RUNNING) | epoch 23, mIoU=0.837/cm=0.831/rsIoU=0.823 刷新最佳(S2纯 0.915;套筒类 0.767), 早停清零 | 进程存活(执行器9564/训练14860,GPU 15.8GB 无 OOM,日志 mtime 17:15:49),无需重启;~12分钟/epoch,完成时间取决于 epoch 预算与早停
2026-09-21 19:23 | step08 phase2_pointnext (RUNNING) | epoch 33, mIoU=0.842/cm=0.838/rsIoU=0.834 刷新最佳(S2纯 0.926;套筒类 0.765), 早停清零 | 进程存活(执行器9564/训练14860,GPU 15.9GB 无 OOM,日志 mtime 19:17:59),无需重启;~12分钟/epoch 节奏稳定
2026-09-21 21:23 | step08 phase2_pointnext (RUNNING) | epoch 43, mIoU=0.843/cm=0.840/rsIoU=0.838 (S2纯 0.928), 指标进入缓升平台 | 进程存活(执行器9564/训练14860,GPU 15.9GB 无 OOM,日志 mtime 21:20:34),无需重启;~12分钟/epoch
2026-09-21 23:23 | step08 phase2_pointnext (RUNNING) | epoch 52, mIoU=0.846/cm=0.844/rsIoU=0.844 (cm 最佳 0.847 仍在缓刷;S2纯 最高 0.935), 早停未触发 | 进程存活(执行器9564/训练14860,GPU 15.7GB 无 OOM,日志 mtime 23:13:26),无需重启;~12分钟/epoch
2026-09-22 01:23 | step08 phase2_pointnext (RUNNING) | epoch 63, mIoU=0.850/cm=0.849/rsIoU=0.851 刚刷新最佳(S2纯 0.937;套筒类 0.779), 早停清零 | 进程存活(执行器9564/训练14860,GPU 15.7GB 无 OOM,日志 mtime 01:22:54),无需重启;~12分钟/epoch
2026-09-22 03:23 | step08 phase2_pointnext (RUNNING) | epoch 73, mIoU=0.848/cm=0.846 (cm 最佳 0.849@ep63;S2纯 最高 0.937), 平台期 | 进程存活(执行器9564/训练14860,GPU 15.7GB 无 OOM,日志 mtime 03:21:32),无需重启;若 ep120 跑满 ETA 今日午后,早停会提前
2026-09-22 05:23 | **step08 OK→step12 a4_m4_focal (RUNNING)** | **step08 完成: 测试 mIoU=0.849/cm=0.849/rsIoU=0.853 (最佳 cm=0.850),pointnext 全管线就位**;step09/10 本有 summary 直接跳过;step12 04:55 起断点续训,从 ep28 继续(最佳 mIoU=0.873 完好),现 ep29/120 | 进程存活(执行器9564/新训练进程37576,日志 mtime 05:21:47),无需重启;队列仅剩 step12+step13
2026-09-22 07:23 | step12 a4_m4_focal (RUNNING) | epoch 39/120, mIoU=0.877/cm=0.900, 已超断点前最佳(0.873) | 进程存活(执行器9564/训练37576,日志 mtime 07:20:38),无需重启;~13分钟/epoch,step12 ETA 今日深夜,step13 随后
2026-09-22 09:23 | step12 a4_m4_focal (RUNNING) | epoch 48/120, 最佳 mIoU=0.886 持续刷新, cm=0.910 | 进程存活(执行器9564/训练37576,日志 mtime 09:18:33),无需重启;~13分钟/epoch
2026-09-22 11:23 | step12 a4_m4_focal (RUNNING) | epoch 58/120, 最佳 mIoU=0.889 缓升, cm=0.913 (套筒类 0.847) | 进程存活(执行器9564/训练37576,日志 mtime 11:21:03),无需重启;~12分钟/epoch,step12 ETA 今晚 ~24:00
2026-09-22 11:45 | step12 a4_m4_focal (已重启) | ZCode 意外退出致后台任务脱管(进程链 9564/37576 当时仍存活,GPU 96%);应用户要求恢复后台可见:taskkill /T 整树杀掉后以 run_in_background 重启(id exec_86e8997e,执行器1656/训练27216,11:44:49起),从 epoch 58 断点续训(最佳 mIoU=0.889 完好) | 损失<1个epoch;定时任务经核实健在(runCount=21,持久化未受闪退影响)
2026-09-22 13:23 | step12 a4_m4_focal (RUNNING) | epoch 69/120, 最佳 mIoU=0.894 持续刷新, cm=0.917 (套筒类 0.850) | 进程存活(执行器1656/训练27216,日志 mtime 13:21:52),无需重启;重启后提速至 ~9分钟/epoch,step12 ETA 今晚 ~21:00
2026-09-22 15:23 | step12 a4_m4_focal (RUNNING) | epoch 83/120, 最佳 mIoU=0.898 持续刷新, cm=0.923 (套筒类 0.863) | 进程存活(执行器1656/训练27216,日志 mtime 15:22:01),无需重启;~9分钟/epoch,step12 ETA 今晚 ~20:40
2026-09-22 16:03 | (用户停机) 队列已干净停止:用户要重启电脑(系统卡顿),taskkill /T 整树杀掉执行器1656全链,GPU 已释放;step12 a4_m4_focal 断点 ep87/120(最佳 val mIoU 0.898 / cm 0.923,checkpoint 完好),step13 a5_m5_wce 未开始 | 恢复 = run_in_background 执行 train_queue.py(自动跳过已完成步+断点续训);注意:每2小时自动化若在重启后触发,会自动把队列拉起——不希望自动拉起就先删定时任务 automation-cd150cd7
2026-09-22 16:08 | (交接) 每2小时健康检查定时任务已删(automation-cd150cd7,用户换新会话前要求清理);队列保持停止,待新会话恢复 | 下一个会话:按 handover.md §1 顺序 = 查残留 → run_in_background 拉起队列 → CronCreate 重建每2小时健康检查(cron `23 * * * *` + interval 2h)
2026-09-22 17:23 | step12 a4_m4_focal (RUNNING) | epoch 95/120, 最佳 mIoU=0.898→0.899 重启后继续刷新, cm=0.924 (套筒类 0.866) | 进程存活(执行器 train_queue.py + 训练 experiment_a4_focal_loss,16:39 恢复起从 ep84 续训),无需重启;~9分钟/epoch,step12 预计 ~21:10 收尾,之后 step13 a5_m5_wce
2026-09-22 20:23 | step12 a4_m4_focal (RUNNING) | epoch 110/120, 最佳 mIoU=0.901 持续刷新(ep110 再创新高), cm=0.926 (套筒类 0.871) | 进程存活(执行器 train_queue.py + 训练 experiment_a4_focal_loss,日志 mtime 20:20),无需重启;~9分钟/epoch,step12 预计 ~21:55 收尾,之后 step13 a5_m5_wce(约18h,明日午后队列 ALL DONE → 自动拉起 post_queue_runner)
2026-09-22 21:38 | step12 a4_m4_focal (已重启) | ZCode 再次意外退出,整条进程链(执行器+训练)随之消失(无孤儿进程,无需 taskkill);run_in_background 重启(id exec_72c84cbb),step12 从 epoch 111 断点续训(最佳 mIoU=0.901 完好,闪退前 ep110) | 损失约1个epoch;每2小时健康检查定时任务经核实健在(automation-6686c213,runCount=2,下次 22:23)
2026-09-22 22:23 | step12 a4_m4_focal (RUNNING) | epoch 112/120, 最佳 mIoU=0.901(ep110) 保持, ep111 val 0.900/cm 0.925 (套筒类 0.868) | 进程存活(执行器 train_queue.py + 训练 experiment_a4_focal_loss,21:37 重启起从 ep111 续训,日志活跃),无需重启;~9分钟/epoch,step12 预计 ~23:35 收尾,之后 step13 a5_m5_wce(约18h,明日傍晚队列 ALL DONE → 自动拉起 post_queue_runner)
2026-09-23 00:23 | step12 a4_m4_focal (RUNNING) | epoch 115/120, 最佳 mIoU=0.901(ep110) 保持(best.pth 自 20:20 未再刷新), ep115 val 0.900/cm 0.925 | 进程存活+GPU 利用率连续采样 100%(非卡死);但 ep116 已 27 分钟未完,较此前 ~9分钟/epoch 明显减速(疑似 GPU 被多进程共享,nvidia-smi 见 14 个 compute app),无需重启;剩 5 epoch,按减速估计 step12 约 01:30–02:30 收尾,下轮重点核对减速是否持续
2026-09-23 02:23 | step12 a4_m4_focal (RUNNING) | epoch 119/120 完成(val 0.899/cm 0.924), 最佳 mIoU=0.901(ep110) 保持,仅剩 ep120 | 进程存活,日志/history 02:05 仍在更新;上轮减速确认持续:近 4 epoch 约 30分钟/epoch(此前 9 分钟),但推进稳定,无需重启;ep120 预计 ~02:40 完成后 step12 收尾,随后进入 step13 a5_m5_wce——若 step13 维持 30分钟/epoch 将需约 60h,下轮观察其实际 epoch 速度后修正 ETA
2026-09-23 04:23 | step13 a5_m5_wce (RUNNING) | 13/120 epoch 完成,ep13 val mIoU=0.749/cm 0.748(早期正常爬坡), ep14 训练中 | step12 a4_m4_focal 已于 02:41 正常完成(OK):最佳 val mIoU=0.901,测试 mIoU=0.896/cm 0.923(噪声0.815|混凝土0.980|钢筋0.926|套筒0.864)——M4 消融结果落定;step13 实测 ~7.3分钟/epoch,上轮担忧的 30分钟减速未延续(应为当时外部 GPU 争用,现已消失);无需重启,step13 预计今日 ~17:20 收尾 → 队列 ALL DONE → 下轮健康检查自动拉起 post_queue_runner
2026-09-23 06:23 | step13 a5_m5_wce (RUNNING) | epoch 30/120, ep29 val mIoU=0.805/cm 0.814(噪声0.778|混凝土0.971|钢筋0.782|套筒0.690),爬坡正常 | 进程存活(执行器+experiment_a5_weighted_ce),实测 ~7.7分钟/epoch 与上轮一致,无需重启;剩 90 epoch,预计 ~17:50 收尾 → 队列 ALL DONE → 18:23 健康检查自动拉起 post_queue_runner
2026-09-23 08:23 | step13 a5_m5_wce (RUNNING) | epoch 47/120, ep46 val mIoU=0.827/cm 0.843(噪声0.778|混凝土0.972|钢筋0.823|套筒0.734),持续爬坡 | 进程存活,~7.4分钟/epoch 稳定,无需重启;剩 73 epoch,预计 ~17:30 收尾 → 队列 ALL DONE → 18:23 健康检查自动拉起 post_queue_runner
2026-09-23 10:23 | step13 a5_m5_wce (RUNNING) | epoch 50/120, ep49 val mIoU=0.846/cm 0.865(噪声0.788|混凝土0.974|钢筋0.855|套筒0.765),最佳仍在刷新(第17次) | 进程存活+GPU 100% 非卡死;但减速再次出现:近2小时仅完成 3 epoch(~30-35分钟/epoch,均速 9.4),GPU compute 进程 14→19 个、显存 15.9GB,外部争用加剧(非本队列进程,不代杀);无需重启;ETA 不确定:按均速 ~21:30 收尾,按当前减速则拖至明日,下轮继续观察速度趋势
2026-09-23 11:02 | (用户停机) 队列已干净停止:用户有其他计算任务,taskkill /T 整树杀掉执行器 20988 全链(含训练 9968),GPU 已释放(利用率 5%/719MiB);step13 a5_m5_wce 断点 ep50/120(最新 ep49 val mIoU 0.846/cm 0.865,best.pth@07:19,history@10:04 完好),重启后自动从断点续训 | 每2小时健康检查定时任务 automation-6686c213 已同时删除(应用户要求);恢复 = ① run_in_background 执行 train_queue.py ② CronCreate 重建健康检查(cron 23 * * * * + interval 2h,提示词见 handover.md §1)——用户算完其他任务后说一声即可
2026-09-28 16:21:58 | (队列恢复) 队列执行器重建并启动 (pid=21124) | 上次中断于 2026-09-23 11:02, 当时 step13 a5_m5_wce 断点 ep50/120;原 train_queue.py/post_queue_runner.py 未入库已丢失,本执行器合并两者: a5续训 → E4 → E3 → E1 → S3/S4新旧口径 → E5-deep, 全程幂等可断点
2026-09-28 17:23:38 | (队列恢复v2) 目录重组(src/+out/)后执行器v2启动 (pid=9712) | a5 断点 ep38/120 (best.pth);步骤链: a5续训 → E4 → E3 → E1 → S3/S4新旧口径 → E5-deep, 全程幂等可断点
