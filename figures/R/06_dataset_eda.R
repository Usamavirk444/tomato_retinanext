# ─────────────────────────────────────────────────────────────────────
#  Figure 1 — dataset analysis (four panels, one file)
#
#  Source: data/clean_yolo_dataset/*/labels/*.txt, read in full.
#
#  Every one of the 67,595 annotations is used. No subsampling, so the
#  counts and percentages here are exact and match Section 2 of the paper.
#  Previously written as three separate files; combined into the single
#  figure the paper prints.
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

boxes <- read_dataset_boxes()
message("  ", format(nrow(boxes), big.mark = ","), " annotations, ",
        length(unique(boxes$class)), " classes")

# exact statistics, printed so they can be checked against Section 2
stats <- list(
  n          = nrow(boxes),
  small      = mean(boxes$area_frac < 0.01) * 100,
  large      = mean(boxes$area_frac > 0.50) * 100,
  centre     = mean(sqrt((boxes$x_centre - .5)^2 + (boxes$y_centre - .5)^2) < .06) * 100)
print(unlist(stats))

# (a) class distribution by split
pA <- boxes %>% count(class, split, name = "n") %>%
  ggplot(aes(reorder(class, n, sum), n, fill = split)) +
  geom_col(width = 0.72) + coord_flip() +
  scale_fill_manual(values = c(train = unname(PAL["blue"]),
                               val   = unname(PAL["teal"]),
                               test  = unname(PAL["grey_mid"]))) +
  scale_y_continuous(labels = comma, expand = expansion(mult = c(0, 0.08))) +
  labs(x = NULL, y = "Annotations", subtitle = "Class distribution by split") +
  theme_paper() + theme(panel.grid.major.y = element_blank())

# (b) box area. Log scale because areas span three orders of magnitude.
pB <- ggplot(boxes, aes(area_frac)) +
  geom_histogram(bins = 60, fill = PAL[["blue"]], colour = "white", linewidth = 0.1) +
  geom_vline(xintercept = c(0.01, 0.5), linetype = "dashed",
             colour = PAL[["red"]], linewidth = 0.4) +
  # One two-line label centred (on the log scale) between the 1% and 50%
  # lines, on a white box so neither dashed line runs through the text.
  annotate("label", x = sqrt(0.01 * 0.5), y = Inf, hjust = 0.5, vjust = 1.25,
           size = 2.2, lineheight = 1.05, colour = PAL[["red"]],
           fill = "white", label.size = 0,
           label.padding = unit(0.12, "lines"),
           label = sprintf("%.1f%% below 1%%\n%.1f%% above 50%%",
                           stats$small, stats$large)) +
  scale_x_log10(labels = percent) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.16))) +
  labs(x = "Box area as a share of the image (log scale)", y = "Count",
       subtitle = "Box size spans three orders of magnitude") +
  theme_paper()

# (c) aspect ratio
pC <- ggplot(boxes, aes(aspect)) +
  geom_histogram(bins = 60, fill = PAL[["teal"]], colour = "white", linewidth = 0.1) +
  geom_vline(xintercept = 1, linetype = "dashed",
             colour = PAL[["grey_dark"]], linewidth = 0.4) +
  scale_x_log10(limits = c(0.1, 10), breaks = c(0.1, 0.25, 0.5, 1, 2, 4, 10)) +
  labs(x = "Width / height (log scale)", y = "Count",
       subtitle = "Aspect ratio; dashed line is square") +
  theme_paper()

# (d) centre density, all boxes binned (not sampled)
pD <- ggplot(boxes, aes(x_centre, y_centre)) +
  geom_bin2d(bins = 60) +
  scale_fill_blues(trans = "log10", labels = comma, name = "Boxes") +
  coord_fixed() + scale_y_reverse() +
  labs(x = "x (normalised)", y = "y (normalised)",
       subtitle = sprintf("Centre density; %.1f%% within 0.06 of the image centre",
                          stats$centre)) +
  theme_paper() + theme(legend.position = "right",
                        panel.grid.major = element_blank())

p <- (pA | pB) / (pC | pD) +
  plot_annotation(
    title = "Dataset composition and annotation geometry",
    subtitle = sprintf("All %s annotations across 24,876 images; no subsampling",
                       format(stats$n, big.mark = ",")),
    tag_levels = "a",
    caption = "Centre clustering reflects tiled augmentation in the source dataset. It is a property of the data, not of any model.",
    theme = theme(plot.title = element_text(face = "bold", size = 12),
                  plot.subtitle = element_text(size = 9, colour = PAL[["grey_dark"]]),
                  plot.caption = element_text(size = 7, colour = PAL[["grey_mid"]], hjust = 0)))

save_fig(p, "Fig01_dataset_analysis", 9.5, 8)
print(p)
