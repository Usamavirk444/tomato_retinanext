# ─────────────────────────────────────────────────────────────────────
#  Ablation — in-domain accuracy against cross-dataset transfer
#
#  Sources: main_results_table.csv (S1, S5), <stage>_seed_<n>_eval_results.csv
#           (S2-S4), cross_eval_3/ and cross_eval_3_ablation/
#
#  Previously two separate figures. The second (generalisation gap) was the
#  same five numbers subtracted, so it is now a right-hand panel sharing the
#  stage axis. Stage order is the architectural build order and is NOT
#  reordered by performance. Both panels use mAP@0.5 only; the 0.5:0.95
#  column is left to the table rather than mixed onto one axis.
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

d <- read_ablation()
print(d)   # check against Table 4 before using the figure

long <- bind_rows(
  d %>% transmute(stage, domain = "In-domain test set", map50 = indomain, sd = indomain_sd),
  d %>% transmute(stage, domain = "External benchmark", map50 = cross,    sd = cross_sd)) %>%
  mutate(domain = factor(domain, levels = names(DOMAIN_COLS)))

pA <- ggplot(long, aes(stage, map50, fill = domain)) +
  geom_col(position = position_dodge(0.76), width = 0.68) +
  geom_errorbar(aes(ymin = map50 - sd, ymax = map50 + sd),
                position = position_dodge(0.76), width = 0.2,
                linewidth = 0.35, colour = PAL[["grey_dark"]]) +
  geom_text(aes(label = sprintf("%.1f", map50)),
            position = position_dodge(0.76), vjust = -0.9, size = 2.6) +
  scale_fill_manual(values = DOMAIN_COLS) +
  scale_y_metric() +
  labs(title = "Ablation: accuracy and transfer are not the same ranking",
       subtitle = "S3 leads in-domain; the complete model leads on the external benchmark",
       x = NULL, y = "mAP@0.5 (%)") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1, size = 8))

pB <- ggplot(d, aes(stage, drop, fill = stage)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.1f", drop)), vjust = -0.6, size = 2.7) +
  scale_fill_manual(values = STAGE_COLS, guide = "none") +
  scale_y_continuous(limits = c(0, 56), breaks = seq(0, 50, 10),
                     expand = expansion(mult = c(0, 0.03))) +
  labs(title = "Generalisation gap",
       subtitle = "In-domain minus external, percentage points. Lower is better.",
       x = NULL, y = "Drop (pp)") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 20, hjust = 1, size = 8))

p <- pA / pB + plot_layout(heights = c(1.25, 1)) +
  plot_annotation(tag_levels = "a",
    caption = "Differences are not tested for significance. Component contributions are measured, not causally attributed.",
    theme = theme(plot.caption = element_text(size = 7, colour = PAL[["grey_mid"]], hjust = 0)))

save_fig(p, "Fig_ablation_indomain_vs_cross", 7.2, 7.4)
print(p)
