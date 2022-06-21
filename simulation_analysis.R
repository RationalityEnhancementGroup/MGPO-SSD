# Read data
df = read.csv("./data/simulation_results/simulation_results.csv")

df$Name <- as.factor(df$Name)
df$EnvType <- as.factor(df$EnvType)
df$Cost <- as.factor(df$Cost)

model <- aov(ExpectedReward ~ Name * EnvType * Cost, data=df)
summary(model)

TukeyHSD(aov(ExpectedReward ~ Name * EnvType * Cost, data=df), "Name")
