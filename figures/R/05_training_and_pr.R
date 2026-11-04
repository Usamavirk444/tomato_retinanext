# ─────────────────────────────────────────────────────────────────────
#  Figure S1 — training convergence   (recommended: supplementary)
#  Figure 7  — measured precision-recall curves
#
#  Sources: source_csv/logs/*/seed_*/\*history.csv
#           source_csv/pr_curves/fig7_pr_curves_REAL_values.csv
#
#  Training curves show SEED 42 ONLY. They are not averaged across runs,
#  and a single run says nothing about reproducibility — the seed spread in
#  Figure 4 is the evidence for that. Recommended for supplementary
#  material: convergence is not part of the paper's argument.
#
#  The PR curves are measured. They come from a TorchMetrics accumulator
#  with extended_summary = TRUE: the 101-point interpolated precision array
#  at IoU 0.50 that COCO averages to produce AP. The area under each curve
#  therefore equals that class's AP@0.5 by construction, which the script
#  checks below.
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

# ── training, seed 42 ────────────────────────────────────────────────
tc <- read_training() %>% filter(seed == 42, !is.na(map50))

pA <- ggplot(tc, aes(epoch, map50, colour = model)) +
  geom_line(linewidth = 0.55) +
  scale_colour_manual(values = STAGE_COLS, guide = guide_legend(nrow = 2)) +
  labs(x = "Epoch", y = "Validation mAP@0.5 (%)", subtitle = "Validation mAP@0.5") +
  theme_paper()

pB <- ggplot(tc %>% filter(!is.na(train_loss)), aes(epoch, train_loss, colour = model)) +
  geom_line(linewidth = 0.55) +
  scale_colour_manual(values = STAGE_COLS, guide = guide_legend(nrow = 2)) +
  labs(x = "Epoch", y = "Training loss", subtitle = "Training loss") +
  theme_paper()

# One shared legend for both panels, placed once below panel (b).
pS1 <- pA / pB + plot_layout(guides = "collect") +
  plot_annotation(title = "Training convergence (seed 42 only)",
    subtitle = "A single run. Run-to-run variation is shown in Figure 4, not here.",
    tag_levels = "a",
    theme = theme(plot.title = element_text(face = "bold", size = 12),
                  plot.subtitle = element_text(size = 9, colour = PAL[["grey_dark"]]))) &
  theme(legend.position = "bottom")

save_fig(pS1, "FigS1_training_curves", 7.2, 6.4)

# ── measured PR curves ───────────────────────────────────────────────
pr <- read_pr_curves()
auc <- pr %>% group_by(class) %>% summarise(ap = mean(precision), .groups = "drop")
message("  AUC of each curve (equals AP@0.5 by construction):")
print(auc, n = 9)

p7 <- ggplot(pr, aes(recall, precision)) +
  geom_area(fill = PAL[["blue"]], alpha = 0.15) +
  geom_line(colour = PAL[["blue"]], linewidth = 0.6) +
  geom_text(data = auc, aes(0.04, 0.08, label = sprintf("AP = %.3f", ap)),
            inherit.aes = FALSE, hjust = 0, size = 2.5, colour = PAL[["grey_dark"]]) +
  facet_wrap(~ class, ncol = 3) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  scale_y_continuous(limits = c(0, 1.02), breaks = seq(0, 1, 0.25)) +
  labs(title = "Per-class precision-recall curves — RetinaNeXt (seed 42)",
       subtitle = "Measured from model predictions at IoU 0.50; 101-point interpolation",
       x = "Recall", y = "Precision",
       caption = "Single seed. Curves are not averaged across runs.") +
  theme_paper() +
  theme(strip.text = element_text(size = 8))

save_fig(p7, "Fig07_pr_curves", 7.2, 6.8)
print(pS1); print(p7)
