library(lme4)
library(lmerTest)
library(readr)

df <- read_csv("data/out/openface_data_long.csv", show_col_types = FALSE)

df$emotion <- factor(df$emotion)
df$condition <- factor(df$condition)
df$video_id <- factor(df$video_id)
df$AU <- factor(df$AU)

df$condition <- relevel(df$condition, ref = "original")


# df_orig <- df[df$condition == "original", ]
# sub <- df_orig[df_orig$AU == "AU12_r_mean", ]
# m <- lmer(activation ~ emotion + (1|video_id), data=sub, REML=FALSE)
# a <- anova(m)
# print(a)
# summary(m)


sub <- df[df$AU == "AU12_r_mean", ]

m <- lmer(activation ~ condition + (1|video_id) + (1|emotion), data=sub, REML=FALSE)
a <- anova(m)
print(a)
summary(m)