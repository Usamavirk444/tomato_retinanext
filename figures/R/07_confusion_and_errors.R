# ─────────────────────────────────────────────────────────────────────
#  Figure 9  — detection confusion matrices (seed 42)
#  Figure 12 — error breakdown
#
#  Source: source_csv/results/<model>/seed_42/confusion_matrix_test.csv
#
#  MATCHING PROTOCOL (paper_files/confusion_matrix.py), state this in the
#  caption: a prediction is matched to the highest-IoU ground-truth box;
#  a match counts when IoU >= 0.50; each ground-truth box can be claimed
#  once; unmatched predictions become false positives; unmatched ground
#  truth becomes a miss. Predictions below confidence 0.05 are discarded.
#
#  CORRECTION MADE HERE. The stored matrix has a tenth row, "background",
#  holding false positives grouped by predicted class. It sums to the total
#  number of false positives, whereas each class row sums to that class's
#  ground-truth count. Row-normalising both and drawing them on one colour
#  scale implies they are comparable; they are not. The background row is
#  therefore excluded from the heatmap and shown separately in panel (b)
#  of Figure 12, on its own denominator.
#
#  NOTE FOR THE CAPTION: this confidence threshold (0.05) differs from the
#  0.01 used for the mAP values in Table 1, so the two describe different
#  operating points.
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

cf <- read_confusion(42)
cm <- cf$cm      # nine class rows, row-normalised -> per-class recall
fp <- cf$fp      # background row, normalised over all false positives

stopifnot(all(abs(
  cm %>% group_by(model, true_class) %>% summarise(s = sum(row_norm), .groups = "drop") %>%
    pull(s) - 1) < 1e-8))

p9 <- ggplot(cm, aes(pred_class, true_class, fill = row_norm)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = ifelse(row_norm >= 0.01, sprintf("%.2f", row_norm), "")),
            size = 1.9, colour = ifelse(cm$row_norm > 0.60, "white", PAL[["grey_dark"]])) +
  facet_wrap(~ model) +
  scale_fill_blues(limits = c(0, 1), labels = percent,
                   breaks = c(0, 0.5, 1), name = "Row share") +
  scale_y_discrete(limits = rev) +
  labs(title = "Detection confusion matrices (seed 42)",
       subtitle = "Rows normalised by ground-truth count, so the diagonal is per-class recall",
       x = "Predicted class", y = "True class",
       caption = paste0("IoU \u2265 0.50, confidence \u2265 0.05, greedy matching.\n",
                        "The 'background' column counts ground-truth objects the model missed.\n",
                        "False positives have a different denominator and are shown in Fig 12b.")) +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 6.5),
        axis.text.y = element_text(size = 6.5),
        legend.position = "right", panel.grid.major = element_blank(),
        legend.key.height = unit(1.6, "lines"),
        plot.caption = element_text(size = 7, hjust = 0, lineheight = 1.2))

save_fig(p9, "Fig09_confusion_matrices", 7.2, 7)

# ── Figure 12 ────────────────────────────────────────────────────────
# (a) per class, the share of ground-truth instances that were missed or
#     assigned the wrong class. Denominator: ground-truth count for that
#     class. These two plus recall sum to 1.
# (b) where false positives land. Denominator: total false positives.
err <- cm %>%
  filter(model == "RetinaNeXt") %>%
  group_by(true_class) %>%
  # Compare as character, not factor. pred_class carries an extra
  # "background" level that true_class does not, and R refuses to compare
  # two factors whose level sets differ.
  summarise(`Missed (no detection)` =
              sum(row_norm[as.character(pred_class) == "background"]),
            `Confused with another class` =
              sum(row_norm[as.character(pred_class) != as.character(true_class) &
                           as.character(pred_class) != "background"]),
            .groups = "drop") %>%
  pivot_longer(-true_class, names_to = "error", values_to = "rate") %>%
  mutate(error = factor(error, levels = names(ERROR_COLS)))

p12a <- ggplot(err, aes(reorder(true_class, rate, sum), rate, fill = error)) +
  geom_col(width = 0.68) + coord_flip() +
  scale_fill_manual(values = ERROR_COLS) +
  scale_y_continuous(labels = percent, expand = expansion(mult = c(0, 0.05))) +
  labs(x = NULL, y = "Share of ground-truth instances",
       subtitle = "Denominator: ground-truth count per class") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank())

p12b <- fp %>% filter(model == "RetinaNeXt") %>%
  ggplot(aes(reorder(pred_class, share), share)) +
  geom_col(width = 0.66, fill = PAL[["blue"]]) +
  geom_text(aes(label = sprintf("%.0f", count)), hjust = -0.2, size = 2.5,
            colour = PAL[["grey_dark"]]) +
  coord_flip() +
  scale_y_continuous(labels = percent, expand = expansion(mult = c(0, 0.12))) +
  labs(x = NULL, y = "Share of all false positives",
       subtitle = "Denominator: total false positives. Counts shown at bar end.") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank())

p12 <- p12a / p12b +
  plot_annotation(
    title = "Error analysis — RetinaNeXt (seed 42)",
    subtitle = "Panels use different denominators and must not be read against one another",
    tag_levels = "a",
    theme = theme(plot.title = element_text(face = "bold", size = 12),
                  plot.subtitle = element_text(size = 9, colour = PAL[["grey_dark"]])))

save_fig(p12, "Fig12_error_analysis", 7.2, 7.2)
print(p9); print(p12)
