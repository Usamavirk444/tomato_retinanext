# ─────────────────────────────────────────────────────────────────────
#  Figure 13 — accuracy against inference speed
#
#  Sources: source_csv/results/fps_benchmark.csv, main_results_table.csv
#
#  Reduced from two figures to one. The previous bar chart of FPS and
#  parameter count restated Table 5 value for value and is dropped; this
#  scatter is the only panel that shows the trade-off itself.
#
#  Measurement conditions, from fps_benchmark.csv: batch size 1,
#  640 x 640 input, CUDA. FPS and latency are internally consistent
#  (FPS = 1000 / latency_ms) and are checked below. These are measured
#  times, not FLOPs.
#
#  FLAGGED: latency standard deviation is 9.71 ms on 30.91 ms for
#  RetinaNet (31%) and 10.30 on 38.26 for RetinaNeXt (27%), against
#  0.24 on 11.98 for YOLOv8n (2%). That pattern suggests warm-up
#  iterations were not discarded for the torchvision models. The means
#  are reported as measured; the variation is worth re-checking.
# ─────────────────────────────────────────────────────────────────────
# Portable resolution of this script's own directory.
# sys.frame(1)$ofile is set when the file is source()d (RStudio "Source",
# run_all.R) but NOT under `Rscript file.R`. The previous fallback was an
# absolute path into tomato_vm_clean/, the read-only research archive, which
# does not exist on other machines and which this script must never write to.
.script_dir <- function() {
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a)) dirname(normalizePath(sub("^--file=", "", a[1]))) else getwd()
}
.here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)),
                  error = function(e) .script_dir())
source(file.path(.here, "00_setup.R"))

sp <- read_speed()
ac <- read_main_results() %>% select(model, map50 = map50_mean, map50_sd = map50_std)
d  <- left_join(sp, ac, by = "model")

# FPS must equal 1000 / latency, or the two were measured differently
chk <- with(d, abs(1000 / time_ms_mean - fps) / fps)
if (any(chk > 0.02)) stop("FPS and latency disagree for: ",
                          paste(d$model[chk > 0.02], collapse = ", "))
message("  FPS/latency consistency OK (max deviation ",
        sprintf("%.3f%%", max(chk) * 100), ")")

p <- ggplot(d, aes(fps, map50)) +
  geom_errorbar(aes(ymin = map50 - map50_sd, ymax = map50 + map50_sd),
                width = 1.2, linewidth = 0.35, colour = PAL[["grey_mid"]]) +
  geom_point(aes(colour = model, size = params_M, shape = model)) +
  geom_text(aes(label = sprintf("%s\n%.1fM, %.1f FPS", model, params_M, fps)),
            vjust = -0.9, hjust = 0.5, size = 2.5, lineheight = 0.95,
            colour = PAL[["grey_dark"]]) +
  scale_colour_manual(values = MODEL_COLS, guide = "none") +
  scale_shape_manual(values = c(16, 17, 15, 18), guide = "none") +
  scale_size_continuous(range = c(3, 10), name = "Parameters (M)") +
  scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 20)) +
  scale_y_continuous(limits = c(74, 102), breaks = seq(75, 100, 5)) +
  labs(title = "Accuracy against inference speed",
       subtitle = "Batch size 1, 640 × 640 input, single NVIDIA A100-SXM4-40GB (CUDA)",
       x = "Throughput (frames per second)", y = "mAP@0.5 (%)",
       caption = paste("Point area is parameter count. Whiskers are the three-seed sd of mAP@0.5.",
                       "Measured wall-clock time, not FLOPs.")) +
  theme_paper() +
  theme(legend.position = "right")

save_fig(p, "Fig13_accuracy_vs_speed", 7.2, 5)
print(p)
