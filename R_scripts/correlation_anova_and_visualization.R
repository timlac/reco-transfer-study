library(tidyverse)
library(readr)
library(patchwork)

# ============================================================
# 1. Descriptive labels for Action Units, Pose, Gaze
# ============================================================

var_labels <- c(
  # AUs
  AU01_r_mean = "AU01 – Inner brow raiser",
  AU02_r_mean = "AU02 – Outer brow raiser",
  AU04_r_mean = "AU04 – Brow lowerer",
  AU05_r_mean = "AU05 – Upper lid raiser",
  AU06_r_mean = "AU06 – Cheek raiser",
  AU07_r_mean = "AU07 – Lid tightener",
  AU09_r_mean = "AU09 – Nose wrinkler",
  AU10_r_mean = "AU10 – Upper lip raiser",
  AU12_r_mean = "AU12 – Lip corner puller",
  AU14_r_mean = "AU14 – Dimpler",
  AU15_r_mean = "AU15 – Lip corner depressor",
  AU17_r_mean = "AU17 – Chin raiser",
  AU20_r_mean = "AU20 – Lip stretcher",
  AU23_r_mean = "AU23 – Lip tightener",
  AU25_r_mean = "AU25 – Lips part",
  AU26_r_mean = "AU26 – Jaw drop",
  AU45_r_mean = "AU45 – Blink",

  # Pose
  pose_Rx_mean = "Pose Rx",
  pose_Ry_mean = "Pose Ry",
  pose_Rz_mean = "Pose Rz",
  pose_Tx_mean = "Pose Tx",
  pose_Ty_mean = "Pose Ty",
  pose_Tz_mean = "Pose Tz",

  # Gaze
  gaze_0_x_mean = "Gaze 0 x",
  gaze_0_y_mean = "Gaze 0 y",
  gaze_0_z_mean = "Gaze 0 z",
  gaze_1_x_mean = "Gaze 1 x",
  gaze_1_y_mean = "Gaze 1 y",
  gaze_1_z_mean = "Gaze 1 z",
  gaze_angle_x_mean = "Gaze angle x",
  gaze_angle_y_mean = "Gaze angle y"
)

# Feature groups
au_features   <- names(var_labels)[grepl("^AU", names(var_labels))]
pose_features <- names(var_labels)[grepl("^pose", names(var_labels))]
gaze_features <- names(var_labels)[grepl("^gaze", names(var_labels))]

# ============================================================
# 2. Load data + rename conditions to manuscript terminology
# ============================================================

df <- read_csv("data/out/openface_data_long_all.csv", show_col_types = FALSE) %>%
  mutate(
    condition = recode(condition,
      original = "Human",
      furhat = "Furhat",
      metahuman = "MetaHuman"
    ),
    # optional: make ordering consistent for plots/tables
    condition = factor(condition, levels = c("Human", "MetaHuman", "Furhat"))
  )

# ============================================================
# 3. Collapse duplicates, compute mean activation per feature
# ============================================================

df_condensed <- df %>%
  filter(condition %in% c("Human", "Furhat", "MetaHuman")) %>%
  summarise(
    activation = mean(activation, na.rm = TRUE),
    .by = c(video_id, emotion, feature, condition)
  ) %>%
  pivot_wider(names_from = condition, values_from = activation)

# ============================================================
# 4. Compute correlations between conditions
# ============================================================

safe_cor <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < 3) return(NA_real_)
  x <- x[ok]; y <- y[ok]
  if (sd(x) == 0 || sd(y) == 0) return(NA_real_)
  unname(cor(x, y))
}

corrs <- df_condensed %>%
  group_by(feature) %>%
  summarise(
    `Human–Furhat`     = safe_cor(Human, Furhat),
    `Human–MetaHuman`  = safe_cor(Human, MetaHuman),
    .groups = "drop"
  )

# ============================================================
# 5. Add descriptive labels
# ============================================================

corrs$label <- var_labels[corrs$feature]
corrs$label[is.na(corrs$label)] <- corrs$feature

# ============================================================
# 6. Split into AU / Pose / Gaze subsets
# ============================================================

corr_AU   <- corrs %>% filter(feature %in% au_features)
corr_pose <- corrs %>% filter(feature %in% pose_features)
corr_gaze <- corrs %>% filter(feature %in% gaze_features)

# ============================================================
# 7. Long format for heatmap plotting
# ============================================================

to_long <- function(df_in) {
  df_in %>%
    select(label, `Human–Furhat`, `Human–MetaHuman`) %>%
    pivot_longer(
      cols = -label,
      names_to = "Comparison",
      values_to = "Correlation"
    )
}

corr_AU_long   <- to_long(corr_AU)
corr_pose_long <- to_long(corr_pose)
corr_gaze_long <- to_long(corr_gaze)

# ============================================================
# 8. Heatmap plotting functions
# ============================================================

plot_heatmap <- function(df_in, title_text) {
  ggplot(df_in, aes(x = Comparison, y = label, fill = Correlation)) +
    geom_tile(color = "white") +
    geom_text(
      aes(label = sprintf("%.2f", Correlation)),
      size = 4.2   # numbers inside tiles
    ) +
    scale_fill_gradient2(
      low = "red",
      mid = "white",
      high = "green",
      midpoint = 0,
      limits = c(-1, 1)
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
      axis.text.y = element_text(size = 14),
      axis.title = element_text(size = 15),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 13),
      panel.grid = element_blank()
    ) +
    labs(title = title_text, x = "", y = "")
}


plot_heatmap_nolegend <- function(df_in, title_text, hide_x = FALSE) {

  x_text <- if (hide_x) {
    element_blank()
  } else {
    element_text(size = 16, angle = 45, hjust = 1)
  }

  ggplot(df_in, aes(x = Comparison, y = label, fill = Correlation)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", Correlation)), size = 4.2) +
    scale_fill_gradient2(
      low = "red", mid = "white", high = "green",
      midpoint = 0, limits = c(-1, 1)
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(size = 18, face = "bold"),
      axis.text.x = x_text,
      axis.text.y = element_text(size = 14),
      axis.title  = element_text(size = 15),
      axis.ticks.x = if (hide_x) element_blank() else element_line(),
      legend.position = "none",
      panel.grid = element_blank()
    ) +
    labs(title = title_text, x = "", y = "")
}
# ============================================================
# 9. Build plots
# ============================================================

p_au   <- plot_heatmap(corr_AU_long, "Action Units")                 # keep legend
p_pose <- plot_heatmap_nolegend(corr_pose_long, "Pose")              # no legend
p_gaze <- plot_heatmap_nolegend(corr_gaze_long, "Gaze", hide_x = TRUE)              # no legend
p_pose_gaze <- p_gaze / p_pose

p_all <- p_pose_gaze | p_au # AU plot taller than pose/gaze

# ============================================================
# 10. Save plots
# ============================================================

dir.create("data/plots/correlations", recursive = TRUE, showWarnings = FALSE)

ggsave("data/plots/correlations/heatmap_AU.png",
       p_au, width = 8, height = 10, dpi = 300)

ggsave("data/plots/correlations/heatmap_pose.png",
       p_pose, width = 7, height = 5, dpi = 300)

ggsave("data/plots/correlations/heatmap_gaze.png",
       p_gaze, width = 7, height = 5, dpi = 300)

ggsave("data/plots/correlations/heatmap_pose_gaze.png",
       p_pose_gaze, width = 8, height = 10, dpi = 300)

ggsave("data/plots/correlations/heatmap_all.png",
	   p_all, width = 15, height = 10, dpi = 300)