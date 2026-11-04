# ─────────────────────────────────────────────────────────────────────
#  Figure 14 — cross-dataset generalisation
#
#  Sources: main_results_table.csv, cross_eval_3/*/per_class_cross.csv
#
#  Panel (a): in-domain against external, per model, with the change in
#  percentage points — not a relative percentage.
#  Panel (b): per-class external AP. Mosaic Virus omitted (no ground truth).
#  The cross-dataset standard deviation is across the three training seeds
#  evaluated on one external benchmark; it is not uncertainty over datasets.
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

ind <- read_main_results() %>% select(model, indomain = map50_mean)
cro <- read_cross_overall() %>%
  group_by(model) %>% summarise(cross = mean(map50), cross_sd = sd(map50), .groups = "drop")

d <- left_join(ind, cro, by = "model") %>%
  mutate(model = factor(model, levels = MODEL_LEVELS), drop = indomain - cross)
print(d)   # check against Table 6

long <- bind_rows(
  d %>% transmute(model, domain = "In-domain test set", map50 = indomain, sd = NA_real_),
  d %>% transmute(model, domain = "External benchmark", map50 = cross,    sd = cross_sd)) %>%
  mutate(domain = factor(domain, levels = names(DOMAIN_COLS)))

pA <- ggplot(long, aes(model, map50, fill = domain)) +
  geom_col(position = position_dodge(0.74), width = 0.66) +
  geom_errorbar(aes(ymin = map50 - sd, ymax = map50 + sd),
                position = position_dodge(0.74), width = 0.18,
                linewidth = 0.35, colour = PAL[["grey_dark"]], na.rm = TRUE) +
  geom_text(aes(label = sprintf("%.1f", map50)),
            position = position_dodge(0.74), vjust = -0.8, size = 2.7) +
  geom_text(data = d, aes(model, indomain + 11, label = sprintf("%.2f pp", -drop)),
            inherit.aes = FALSE, size = 2.8, fontface = "bold",
            colour = PAL[["red"]]) +
  scale_fill_manual(values = DOMAIN_COLS) +
  scale_y_metric(limit = 112) +
  labs(x = NULL, y = "mAP@0.5 (%)",
       subtitle = "Change shown in percentage points, not relative change") +
  theme_paper() +
  theme(axis.text.x = element_text(angle = 12, hjust = 1, size = 8))

pc <- read_per_class_cross() %>%
  group_by(model, class) %>%
  summarise(mean = mean(ap50), sd = sd(ap50), .groups = "drop") %>%
  filter(!is.na(mean)) %>%   # Mosaic Virus: no external ground truth
  mutate(model = factor(model, levels = MODEL_LEVELS))

pB <- ggplot(pc, aes(class, mean, fill = model)) +
  geom_col(position = position_dodge(0.78), width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = MODEL_COLS, guide = guide_legend(nrow = 2, reverse = TRUE)) +
  scale_y_metric() +
  labs(x = NULL, y = "External-benchmark AP@0.5 (%)",
       subtitle = "Per class, three-seed mean") +
  theme_paper() +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(colour = PAL[["grey_light"]], linewidth = 0.3))

p <- pA / pB + plot_layout(heights = c(1, 1.3)) +
  plot_annotation(
    title = "Cross-dataset generalisation",
    subtitle = "Filtered external benchmark (805 images, 8 classes with ground truth), evaluated without fine-tuning",
    tag_levels = "a",
    caption = "One external benchmark. Results do not establish generalisation to all field conditions.",
    theme = theme(plot.title = element_text(face = "bold", size = 12),
                  plot.subtitle = element_text(size = 9, colour = PAL[["grey_dark"]]),
                  plot.caption = element_text(size = 7, colour = PAL[["grey_mid"]], hjust = 0)))

save_fig(p, "Fig14_cross_dataset", 7.2, 8)
print(p)
