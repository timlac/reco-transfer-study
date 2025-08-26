# install.packages(c("readr","lme4","lmerTest","emmeans"))  # run once if needed

library(readr)
library(lme4)
library(lmerTest)
library(emmeans)
library(dplyr)

# ==== Load ====
df <- read_csv("data/out/openface_data_long.csv", show_col_types = FALSE)

# Make factors
df$condition <- factor(df$condition)
df$video_id  <- factor(df$video_id)
df$emotion   <- factor(df$emotion)
df$AU        <- factor(df$AU)

df$condition <- relevel(df$condition, ref = "original")

# ==== Mixed model ====
# m1 <- lmer(activation ~ condition + (1 | video_id) + (1 | AU) + (1 | emotion),
#            data = df, REML = FALSE)
#
# summary(m1)    # fixed effects estimates
# anova(m1)      # test of condition effect

# ==== Post-hoc (if condition significant) ====
# emmeans(m1, pairwise ~ condition, adjust = "tukey")
#
# m2 <- lmer(activation ~ condition + (1|video_id) + (1 + condition | AU) + (1|emotion),
#            data = df, REML = FALSE)
#
# summary(m2)    # fixed effects estimates
# anova(m2)      # test of condition effect
#
# # ==== Post-hoc (if condition significant) ====
# emmeans(m2, pairwise ~ condition, adjust = "tukey")


m3 <- lmer(activation ~ condition * AU + (1|video_id) + (1|emotion),
           data = df, REML = FALSE)

summary(m3)    # fixed effects estimates
anova(m3)      # test of condition effect



# m4 <- lmer(activation ~ condition * emotion + (1|video_id) + (1|AU),
#            data = df, REML = FALSE)
#
# summary(m4)    # fixed effects estimates
# anova(m4)      # test of condition effect



# Mixed model: does activation vary by AU, by emotion, and their interaction?
# m_orig <- lmer(activation ~ emotion * AU + (1 | video_id),
#                data = df_orig,
#                REML = FALSE)
#
# summary(m_orig)   # fixed effects estimates
# anova(m_orig)     # test of AU, emotion, and interaction