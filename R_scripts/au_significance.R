library(lme4)
library(lmerTest)
library(readr)

df <- read_csv("data/out/openface_data_long.csv", show_col_types = FALSE)

df_orig <- df[df$condition == "original", ]
df_orig$emotion  <- factor(df_orig$emotion)
df_orig$video_id <- factor(df_orig$video_id)
df_orig$AU       <- factor(df_orig$AU)

au_levels <- unique(df_orig$AU)

results <- lapply(au_levels, function(au) {
  sub <- df_orig[df_orig$AU == au, ]

  m <- lmer(activation ~ emotion + (1|video_id), data=sub, REML=FALSE)
  a <- anova(m)

  data.frame(
    AU = au,
    F_value = a["emotion","F value"],
    p_value = a["emotion","Pr(>F)"]
  )
})

results_df <- do.call(rbind, results)
results_df <- results_df[order(results_df$p_value), ]
print(results_df)
