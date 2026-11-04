# ─────────────────────────────────────────────────────────────────────
#  Figure 4 — main model comparison, carrying everything Table 1 held
#
#  Sources: source_csv/results/main_results_table.csv
#           source_csv/results/<model>/seed_42/confusion_matrix_test.csv
#
#  History of this figure:
#    Fig 4 and the former Fig 5 showed the same quantity twice — mean
#    mAP@0.5 per model, then the per-seed spread of those same numbers.
#    They were merged.
#
#    Table 1 then duplicated the merged figure's two mAP columns, while its
#    FPS and Params columns duplicate Fig 8 (position and point area there).
#    Only Precision, Recall and F1 were unique to the table, so they are
#    folded in here and the table becomes redundant.
#
#    The grouped bar chart was replaced by a dumbbell plot: one row per
#    metric, one dot per model, a connector spanning the range. Bars forced
#    a zero baseline that compressed every difference into the top fifth of
#    the panel; the dumbbell uses the range the data actually occupies and
#    makes the ordering readable at a glance.
#
#  TWO DIFFERENT KINDS OF NUMBER ARE PLOTTED HERE — the caption must say so.
#    mAP@0.5 and mAP@0.5:0.95 are threshold-free, averaged over seeds
#      42/123/456, and carry a standard deviation.
#    Precision, Recall and F1 are computed from the stored confusion
#      matrices at IoU >= 0.50 and confidence >= 0.05, seed 42 only. They
#      are operating-point metrics: raising the confidence threshold raises
#      precision and lowers recall for every model. They are NOT the values
#      printed in the old Table 1, which used a lower threshold and are not
#      reproducible from this folder.
#
#  RetinaNeXt leads four of the five metrics. YOLOv8n has higher precision
#  (0.800 against 0.715) at this threshold. That is shown, not hidden.
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

METRIC_LEVELS <- c("mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1")

# ── threshold-free accuracy, three seeds ─────────────────────────────
mr <- read_main_results()

acc <- mr %>%
  select(model, map50_mean, map50_std, map5095_mean, map5095_std) %>%
  pivot_longer(-model, names_to = c("metric", ".value"),
               names_pattern = "(map50|map5095)_(mean|std)") %>%
  mutate(metric = recode(metric, map50 = "mAP@0.5", map5095 = "mAP@0.5:0.95"))

# ── operating-point metrics, seed 42, IoU 0.50, confidence 0.05 ──────
prf <- read_detection_prf(42) %>%
  select(model, Precision, Recall, F1) %>%
  pivot_longer(-model, names_to = "metric", values_to = "mean") %>%
  mutate(mean = mean * 100, std = NA_real_)

d <- bind_rows(acc, prf) %>%
  mutate(metric = factor(metric, levels = rev(METRIC_LEVELS)),
         model  = factor(model,  levels = MODEL_LEVELS))

# connector spanning the four models within each metric
span <- d %>% group_by(metric) %>%
  summarise(lo = min(mean), hi = max(mean), .groups = "drop")

# leader label, placed to the right of the highest dot in each row
best <- d %>% group_by(metric) %>% slice_max(mean, n = 1) %>% ungroup()

p <- ggplot() +
  geom_segment(data = span, aes(x = lo, xend = hi, y = metric, yend = metric),
               colour = PAL[["grey_light"]], linewidth = 2.6,
               lineend = "round") +
  geom_errorbarh(data = filter(d, !is.na(std)),
                 aes(y = metric, xmin = mean - std, xmax = mean + std),
                 height = 0, linewidth = 0.6, colour = PAL[["grey_dark"]]) +
  geom_point(data = d, aes(mean, metric, fill = model),
             shape = 21, size = 3.4, stroke = 0.7, colour = "white") +
  geom_text(data = best, aes(mean + 1.4, metric, label = sprintf("%.1f", mean),
                             colour = model),
            hjust = 0, size = 2.9, fontface = "bold", show.legend = FALSE) +
  scale_fill_manual(values = MODEL_COLS, name = NULL,
                    guide = guide_legend(nrow = 2, override.aes = list(size = 3.6))) +
  scale_colour_manual(values = MODEL_COLS) +
  scale_x_continuous(limits = c(50, 103), breaks = seq(50, 100, 10),
                     expand = expansion(mult = c(0.01, 0.02))) +
  labs(subtitle = paste("mAP: mean over seeds 42, 123 and 456; horizontal bars are the",
                        "standard deviation.\nPrecision, recall and F1: seed 42, IoU ≥ 0.50,",
                        "confidence ≥ 0.05."),
       x = "Score (%)", y = NULL) +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        legend.position = "bottom")

save_fig(p, "Fig04_main_results", 7.2, 4.4)
print(p)

# Console check: the numbers that replace Table 1.
print(read_detection_prf(42) %>%
        mutate(across(c(Precision, Recall, F1), ~ round(.x, 3))))
