library(lme4)
library(lmerTest)
library(readr)

df <- read_csv("data/out/openface_data_long_all.csv", show_col_types = FALSE)

df_orig <- df[df$condition == "original", ]
df_orig$emotion  <- factor(df_orig$emotion)
df_orig$video_id <- factor(df_orig$video_id)
df_orig$feature       <- factor(df_orig$feature)

feature_levels <- unique(df_orig$feature)

results <- lapply(feature_levels, function(feat) {
  sub <- df_orig[df_orig$feature == feat, ]

  m <- lmer(activation ~ emotion + (1|video_id), data=sub, REML=FALSE)
  a <- anova(m)

  data.frame(
    Feature = feat,
    F_value = a["emotion","F value"],
    p_value = a["emotion","Pr(>F)"]
  )
})

results_df <- do.call(rbind, results)
results_df <- results_df[order(results_df$p_value), ]
print(results_df)
