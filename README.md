# Fantasy Football Regression Model - Using Team and Player Stats to Improve Fantasy Team Performance

## Introduction
In Fantasy Football, oftentimes, when deciding on whom to draft for a fantasy football team, users focus on their own team and player preference (i.e. their favorite teams and favorite players) and occasionally individual players past year’s performance. The problem here is that only the players past year’s performance is quantifiable, and emotional decisions like team and player preference do not have a quantifiable reasoning behind them, which can easily lead to poor overall fantasy team performance. In addition to this, users then miss out on otherwise extremely valuable insights, such as the past performance of the actual team that the individual players belong to (after all, football is a team sport), how various off-season trades and NFL draft picks will influence the performance of that team in the next year, and the expected difficulty of a teams schedule in the coming season (i.e. a receiver on a team that is playing a lot of other teams with a great passing defenses will have a more difficult time scoring fantasy points than a receiver on a team that is more often playing teams with poor passing defenses). This project addresses some of these issue by using both player and team stats and the upcoming season’s schedule to determine which teams are the best teams to draft players from.

## The Data
In this project, I make use of the freely available NFL team, game, and player [datasets](https://github.com/ryurko/nflscrapR-data) from the [nflscrapR package](https://github.com/maksimhorowitz/nflscrapR) - which contains hundreds of thousands of rows and hundreds of columns. From this package, I specifically made use of the following datasets at some point or another throughout this project:
- All pre season, regular season, and post season [play-by-play datasets](https://github.com/ryurko/nflscrapR-data/tree/master/play_by_play_data) from 2009 to 2018
- All pre season, regular season, and post season [game datasets](https://github.com/ryurko/nflscrapR-data/tree/master/games_data) from 2009 to 2018
- All [game_player_stats datasets](https://github.com/ryurko/nflscrapR-data/tree/master/legacy_data/game_player_stats)
- All [game_team_stats datasets](https://github.com/ryurko/nflscrapR-data/tree/master/legacy_data/game_team_stats)
- All [season_player_stats datasets](https://github.com/ryurko/nflscrapR-data/tree/master/legacy_data/season_player_stats)
- All [season_team_stats datasets](https://github.com/ryurko/nflscrapR-data/tree/master/legacy_data/season_team_stats)

Additionally, I made use of professional football analyst data available with a subscription from [Pro Football Focus (PFF)](https://www.profootballfocus.com/) to build some of my models. From PFF, I used the following datasets:
- General, Rushing, and Passing defensive team statistics datasets
- Defensive player statistics datasets for the following positions:
  - Defensive Interior
  - Defensive Edge
  - Linebacker
  - Safety
  - Cornerback

## Dependencies
Packages and libraries I made use of throughout this project include:
- `pandas`
- `numpy`
- `re`
- `seaborn`
- `matplotlib`
- `py2r`
- `sklearn`

## Data Cleaning
Data from [PFF](https://www.profootballfocus.com/) came very clean from the get-go and didn't require any further cleaning for my initial needs. `nflscrapR` data, however, did take some considerable cleaning. My entire process, which is thoroughly explained, can be seen in the [Fantasy Football Regression Model - nflscrapR data cleaning]() Jupyter notebook, and the highlights are below:
 
To start, since this dataset contains so many potential features, with some datasets containing over 250 columns, I decided to begin by removing any column that was composed of 80% or greater null values straight off the bat, which managed to get rid of over 80 columns from some of the datasets:  

```data.dropna(axis = 1, thresh = int(0.2 * data.shape[0]), inplace = True)```

From there I took a look at all of the remaining column names in each dataset and determined which ones I knew would not ultimately be part of my model, and I dropped these. This process reduced my number of columns down to closer to 100 in the wider play-by-play datasets. After that I began to dig further into the data. I checked the number of unique team names in each of the datasets to ensure that they aligned with the 32 in the NFL, and I realized that this ligned up well for the pre and regular season datasets, but not for the post season datasets due to not every team being present in the post season. For this reason, I chose not to continue with using the post season play-by-play or games datasets that I obtained from `nflsrcapR` because I didn't want the post season data to bias the model. I then took a look at the team names and realized that there were some inconsistencies (i.e. the St Louis Rams and San Diego Chargers moving to LA, thus changing their acronyms from STL and SD to LA and LAC, respectively, and the Jacksonville Jaguars being listed as JAC in some instances and JAX in others), and so went through and corrected these:

```
for key, value in data_dict.items():
    value['home_team'] = value['home_team'].str.replace("JAC", "JAX")
    value['away_team'] = value['away_team'].str.replace("JAC", "JAX")
    value['home_team'] = value['home_team'].str.replace("SD", "LAC")
    value['away_team'] = value['away_team'].str.replace("SD", "LAC")
    value['home_team'] = value['home_team'].str.replace("STL", "LA")
    value['away_team'] = value['away_team'].str.replace("STL", "LA")
    if 'pbp' in key:
        value['posteam'] = value['posteam'].str.replace("JAC", "JAX")
        value['defteam'] = value['defteam'].str.replace("JAC", "JAX")
        value['posteam'] = value['posteam'].str.replace("SD", "LAC")
        value['defteam'] = value['defteam'].str.replace("SD", "LAC")
        value['posteam'] = value['posteam'].str.replace("STL", "LA")
        value['defteam'] = value['defteam'].str.replace("STL", "LA")
```

At that point I felt that the play-by-play as well as games datasets were cleaned, so I wrote them all out to CSV files. 

Next I began looking into the other datasets, like the team and player `game_recieving`, `game_passing`, and `game_rushing` datasets; the team and player `season_receiving`, `season_passing`, and `season_rushing` datasets; as well as the `season_def_team_receiving`, `season_def_team_passing` and `season_def_team_rushing` datasets. I made some comparisons between these in order to understand the differences and found that the `team` datasets really only focused on MVPs from each game, so I opted not to use these in the rest of my analysis and only continued cleaning the `game_player_xxx` and `season_player_xxx` datasets. I examined column names and found some inconsistencies in the naming conventions between these and the play-by-play datasets, such as there being `Receiver_ID`,`Passer_ID` and `Rusher_ID` columns rather than just `player_id` columns, many columns were in camel case instead of snake case, and in some places `touchdown` columns were listed as `TD` or `TDs`, so I reconsiled these: 

```
to_rename = {"GameID": "game_id"}
game_player_receiving.rename(columns = to_rename, inplace = True)
game_player_rushing.rename(columns = to_rename, inplace = True)
game_player_passing.rename(columns = to_rename, inplace = True)

game_player_receiving.rename(columns = {"Receiver_ID": "player_id"}, inplace = True)
game_player_rushing.rename(columns = {"Rusher_ID": "player_id"}, inplace = True)
game_player_passing.rename(columns = {"Passer_ID": "player_id"}, inplace = True)

# Code adapted from: https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def convert(name):
    name = name.replace("TDs", "touchdowns")
    name = name.replace("TD", "touchdown")
    s1 = first_cap_re.sub(r'\1_\2', name)
    s1 = all_cap_re.sub(r'\1_\2', s1).lower()
    return s1.replace("__", "_")

game_player_receiving.columns = game_player_receiving.columns.to_series().apply(convert)
game_player_rushing.columns = game_player_rushing.columns.to_series().apply(convert)
game_player_passing.columns = game_player_passing.columns.to_series().apply(convert)

season_player_receiving.rename(columns = {"Receiver_ID": "player_id"}, inplace = True)
season_player_rushing.rename(columns = {"Rusher_ID": "player_id"}, inplace = True)
season_player_passing.rename(columns = {"Passer_ID": "player_id"}, inplace = True)

season_player_receiving.columns = season_player_receiving.columns.to_series().apply(convert)
season_player_rushing.columns = season_player_rushing.columns.to_series().apply(convert)
season_player_passing.columns = season_player_passing.columns.to_series().apply(convert)
```
And with that I felt happy with the data cleaning and wrote my files out to CSVs.

## EDA (and a bit more cleaning)
After cleaning my data I moved on to Exploratory Data Analysis, or EDA. Here, I explored both the `nflscrapR` data as well as the `PFF` data. __NOTE:__ Since `PFF` data requires a subscription to access, I have redacted this exploration from the [Fantasy Football Regression Model - EDA]() notebook, but I have included some of the code in my explanations here. 

When it comes to EDA, it helps to know what exactly it is we're trying to predict before performing EDA, otherwise we just end up with all sorts of correlation coefficients and no meaningful way to combine these in order to predict a singluar outcome. Fortunately, when it comes to fantasy football, the things that lead to points earned are fairly well established. Personally, I do most of my fantasy footballing with ESPN, so I will used their pointing system. The following list corresponds with [ESPN's Standard Scoring System](http://www.espn.com/fantasy/football/ffl/story?page=fflrulesstandardscoring):

> **Quarterbacks (QB), Running Backs (RB), Wide Receivers (WR), Tight Ends (TE)**  
- 6 pts per rushing or receiving TD
- 6 pts for player returning kick/punt for TD
- 6 pts for player returning or recovering a fumble for TD
- 4 pts per passing TD
- 2 pts per rushing or receiving 2 pt conversion (note: teams do not receive points for yardage gained during the conversion)
- 2 pts per passing 2 pt conversion
- 1 pt per 10 yards rushing or receiving
- 1 pt per 25 yards passing  
*Bonus Points* (note: the player must score a touchdown to score the points)  
- 2 pts per rushing or receiving TD of 40 yards or more
- 2 pts per passing TD of 40 yards or more  
*Penalty Points*    
- -2 pts per intercepted pass
- -2 pts per fumble lost  

> **Kickers (K)**
- 5 pts per 50+ yard FG made
- 4 pts per 40-49 yard FG made
- 3 pts per FG made, 39 yards or less
- 2 pts per rushing, passing, or receiving 2 pt conversion
- 1 pt per Extra Point made  
*Penalty Points*  
- -2 pts per missed FG (0-39 yds)
- -1 pt per missed FG (40-49 yds)  
(note: a missed FG includes any attempt that is blocked, deflected, etc.)  

> **Defensive/Special Teams (D)**  
- 3 pts per defensive or special teams TD
- 2 pts per interception
- 2 pts per fumble recovery (Note: includes a fumble by the opposing team out of the end zone)
- 2 pts per blocked punt, PAT, or FG (Note: a deflected kick of any kind does not receive points)
- 2 pts per safety
- 1 pt per sack

In the fantasy leagues that I'm personally part of, we rarely use Defensive/Special Teams as well as Kickers, so I focused my analysis on the common offensive players: Quarterbacks (QB), Running Backs (RB), Wide Receivers (WR), and Tight Ends (TE). In this way, the variables I focused on predicting are all touchdowns (TDs); rushing, passing, and receiving yards; as well as interceptions and fumbles. 

Now, these datasets have all of these values contained in them (TDs, passings yards, yards after catch \[receiving yards\], rushing yards, interceptions, and fumbles), and I could have relatively easily assumed that past behavior is the best predictor of future behavior and simply done regression analysis on what teams do the best at this versus which do the worst and left my model at that. What I was more insterested in, however, was how effective defensive teams are at ___stopping___ the positive scoring events and ___producing___ the negative scoring events. This allows me to determine how difficult a team is to score points _against_, thus allowing me to determine the relative difficulty of a teams schedule and ultimately result in my ability to choose players from teams who typically do well in terms of producing events that score well in fantasy but who also are playing teams that typically do worse at preventing the positively scoring events/causing the negatively scoring events. 

So, my goal was to find a way to score teams based on their ability to stop touchdowns, limit yards of all types, produce interceptions, and produce fumbles, and this was the basis of my model. Starting with the `nflscrapR` datasets, I grabbed just the columns that relate to these: 
- From the `games` datasets, I grabbed the `home_team`, `away_team`, `home_score`, and `away_score` columns. 
- From the `pbp` datasets, I grabbed the `home_team`, `away_team`, `posteam`, `defteam`, `drive`, `yards_gained`, `air_yards`, `yards_after_catch`, `interception`, `touchdown`, `pass_touchdown`, `rush_touchdown`, `fumble`, `total_home_score`, `total_away_score`, `posteam_score`, `defteam_score`, `posteam_score_post`, and `defteam_score_post`. 
- From both the game_player_receiving and game_player_rushing datasets, I kept `team`, `opponent`, `total_yards`, `fumbles`, and `touchdowns`
- From the game_player_passing dataset, I kept `team`, `opponent`, `attempts`, `completions`, `total_yards`, `touchdowns`, and `air_touchdowns`

The `season_player` dataset only has data on offensive players and didn't tell me a whole lot about the defensive teams, which is what I'm currently trying to score, so I opted to simply ignore these datasets: `season_player_receiving`, `season_player_rushing`, and `season_player_passing`.

After taking care of all this, I was left with consistent columns accross all of my play-by-play and games `nflscrapR` datasets from each year, so I combined all of these into just two larger datasets, each with a column indicating the dataset that this information initially came from.

With this taken care of, I then began down the path of exploring my data. The first thing I wanted to do was get all of the various features into a common frame of reference, so I scaled all of the numeric columns in each of my datasets except for those that were binary using the following User Defined Function (UDF):

```from sklearn import preprocessing

def scale_cols(df, cols, dropna = True):
    df_copy = df.copy()  # using a copy so changes aren't applied in place for debugging purposes
    if dropna:
        # NA values cause problems with scaling and StandardScaler has no way of just ignoring these values
        # so drop rows with NA values before scaling. They are few and far between so shouldn't be an issue
        df_copy.dropna(axis = 0, subset = cols,inplace = True)
        # have to reset the index otherwise later combining becomes an issue
        df_copy.reset_index(drop = True, inplace = True)
    temp = df_copy[cols]
    cols = [col + "_scaled" for col in cols]
    scaler = preprocessing.StandardScaler()
    scaled_df = scaler.fit_transform(temp)
    scaled_df = pd.DataFrame(scaled_df, columns=cols)
    df_copy[cols] = scaled_df
    return df_copy
```
An important note here is that I still had some null values in the `nflscrapR` play-by-play dataset, and as you can tell my `scale_cols` UDF dropps any rows that contained null values. In all columns except for `air_yards` and `yards_after_catch`, null values were only present in 3.5% of well over 500,000 rows, so I simply applied this to all columns except `air_yards`  and `yards_after_catch`. For these 'problem' columns, I took care of scaling by applying a different UDF that followed essentially the same formula as is used in the `preprocessing.StandardScaler` function from `sklearn`:
```
def scale_prob_cols(df, cols):
    for col in cols:
        mean = df[col].mean()
        sd = df[col].std()
        df[col + '_scaled'] = (df[col] - mean) / sd
```

With the scaling taken care of, I began further exploring my dataset by visualizing the numeric columns in each using the `distplot` function from the `seaborn` package. I wanted to see the distributions of multiple numeric columns in each individual dataset on the same plot, so I created another UDF:
```
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


def multivariable_distplot(df):
    plt.figure(figsize=(30, 10))
    for col in df.columns:
        # only process cols of numeric values that have been scaled
        if str(df[col].dtype) == 'object' or not col.endswith('scaled'):
            continue
        print("Processing column: " + str(col))
        sns.distplot(df[[col]], label = col, hist = False, rug = True)
        # add legend with increased size since otherwise too small to easily read
        plt.legend(prop = {'size': 20})
```
Below are some examples of these distribution plots:
#### Games dataset multivariable_distplot:
![](visualizations/distplots/home%20score%20vs%20away%20score%20distplot.png)

#### Defensive Rushing multivariable_distplot:
![](visualizations/distplots/defensive%20rushing%20distplot.png)

#### Safety Statistics multivariable_distplot:
![](visualizations/distplots/safeties%20distplot.png)

These visualizations helped to give me an idea of the distributions of my features, how these distributions compared to one another, and helped to give me ideas for which features might cause issues for a regression model versus which might be good candidates, since regression models typically assume normal distributions.

After that I wanted to get a better idea of how the features in each dataset related to eachother since regression models are sensitive to multicollinearity. So, I decided to generate both correlation matrices and scatter plots from the scaled, numeric columns of each dataset using the following UDF:

```
from pandas.plotting import scatter_matrix
def scatter_matrix_scaled_only(df):
    scaled_cols = [x for x in list(df.columns) if x.endswith('scaled')]
    temp = df[scaled_cols]
    display(temp.corr())
    scatter_matrix(temp, figsize = (15, 15))
```
Below is an example of the `scatter_matrix_scaled_only` output from the `safeties` dataset:
![](visualizations/scatter_matrices/cornerbacks%20correlations%20table.png)
![](visualizations/scatter_matrices/cornerbacks%20scatter%20matrix.png)

From these `scatter_matrix_scaled_only` outputs, I took note of and analyzed the correlation strengths, giving my interpretations of all strong and medium correlations. This helped to give me an idea of what variables I should and should not include in my final models in order to avoid multicolinearity that could easily bias my final model.

## Model Building
After cleaning and exploring my data, I began working on the model itself. Since the datasets I made use of are disparate and there wasn't a good way to combine them column-wise, I decided to create a number of separate models and amalgamate values on a team basis at the end. Then, use those values as well as the NFL schedule for the upcoming season to determine which teams will have the easiest versus most difficult seasons. This way, I can select players from teams with easy seasons and avoid players on teams with difficult seasons, plus having each game forecasted would allow me to know when to trade various players in order to reach the best outcomes. In order to facilitate this, I set out to find defensive teams that produce the most interceptions and lost fumbles while preventing TDs and yards. Knowing this, I decided that my independent variable at its core would be team while my dependent variables would be points, yards, TDs, interceptions, and fumbles lost. Also, since many of the variables use different units, I decided that I should exclusively use the scaled values in my models except for in instances of logistic regression. With all of this in mind, I went through my datasets and found columns that could relate to these variables, and the following table is what I ultimately came up with:

|Dataset |      IV(s)         |                          DVs                                                                           |
|--------|--------------------|--------------------------------------------------------------------------------------------------------|
|def_gen |         TEAM       | YDS_scaled; PTS_scaled; PASS_scaled; RUSH_scaled                                                       | 
|def_pass|         TEAM       | YDS_scaled; TD_scaled; INT_scaled                                                                      |
|def_rush|         TEAM       | YDS_scaled; TD_scaled; FUML_scaled                                                                     |
|cb      |         TEAM       | OVERALL_scaled; COVERAGE_scaled; RUN_DEF_scaled; RUSH_scaled                                           | 
|di      |         TEAM       | OVERALL_scaled; RUN_DEF_scaled; PASS_RUSH_scaled                                                       |
|edge    |         TEAM       | OVERALL_scaled; COVERAGE_scaled; RUN_DEF_scaled; PASS_RUSH_scaled                                      |
|s       |         TEAM       | OVERALL_scaled; COVERAGE_scaled; RUN_DEF_scaled; PASS_RUSH_scaled                                      |
|lb      |         TEAM       | OVERALL_scaled; COVERAGE_scaled; RUN_DEF_scaled; PASS_RUSH_scaled                                      |
|games   |home_team; away_team| home_score_scaled; away_score_scaled                                                                   |
|pbp     |  posteam; defteam  | yards_gained_scaled; air_yards_scaled; yards_after_catch_scaled; interceptions; fumbles; touchdowns    |

With all of that decided, it was time to start building my models. At this point I looked into a number of Python packages for regression, such as `scikit-learn` and `statsmodels`, but all of the implementations seemed unnecessarily complex and required further transformations of my data in order to get it to work. At this point I thought back to how simple linear models are in R, and I decided that I wanted to figure out how to get an R kernel going in my Python jupyter notebook, at which time I found the package `rpy2`. Loading this package allowed me to convert lines of code in a cell to R code using the 'magic' `%R`. So, I created some R cells and read my data into the R kernel:

```
# %R is the 'magic' to use in order to have the subsequent line of code processed by the R kernel
%R pbp = read.csv(file="data/nflscrapR-data/scaled_data/pbp_scaled.csv", header=TRUE, sep=",")
%R games = read.csv(file="data/nflscrapR-data/scaled_data/games_scaled.csv", header=TRUE, sep=",")
%R def_gen = read.csv('data/pff-data/scaled_data/def_gen_scaled.csv', header=TRUE, sep=",")
%R def_pass = read.csv('data/pff-data/scaled_data/def_pass_scaled.csv', header=TRUE, sep=",")
%R def_rush = read.csv('data/pff-data/scaled_data/def_rush_scaled.csv', header=TRUE, sep=",")
%R cb = read.csv('data/pff-data/scaled_data/cb_scaled.csv', header=TRUE, sep=",")
%R di = read.csv('data/pff-data/scaled_data/di_scaled.csv', header=TRUE, sep=",")
%R edge = read.csv('data/pff-data/scaled_data/edge_scaled.csv', header=TRUE, sep=",")
%R lb = read.csv('data/pff-data/scaled_data/lb_scaled.csv', header=TRUE, sep=",")
%R s = read.csv('data/pff-data/scaled_data/s_scaled.csv', header=TRUE, sep=",")
```

From there I began building my models using the `lm()` function. Since I had already decided that I wanted to amalgamate the scores from each model into a final score, I decided that what I really needed from each model was the coefficients of the regression model by team, so after creating each model I grabbed just the coefficients and placed them into a dataframe with the team names as the row names. From there, I knew that I ultimately wanted all of my data back in my Python kernel, so I took the resulting coefficients dataframe and assigned it back to a python dataframe. The overall process for each of my models looked like this: 

```
%R model = lm(cbind(home_score_scaled, away_score_scaled) ~ home_team + away_team, data = games)
%R coefs = model$coefficients
%R df = as.data.frame(coefs, keep.rownames=TRUE)
scores_by_team_model = %R df
```
When it came time to do this with the positional groups datasets (i.e. safeties, defensive interiors, etc.) I realized a number of things. First off, I realized that some of the players in each of these positional groups didn't actually play all that often, and as was noticed in my EDA notebook, the number of snaps a player is part of is correlated with their overall score, indicating that players with low overall scores typically didn't play often, thus these low scores could unfairly bias my model when those players don't actually play much at all. To address this, I decided to drop rows from the positional groups datasets in which the `TOTAL_SNAPS` was less than the average number of `TOTAL_SNAPS` from that dataset. Additionally, I realized that what I cared most about from these datsets were just the `OVERALL` scores for each player as I could use these to get a sense of the overall strength of a defensive team, so I dropped all other columns and combined these datasets into a single `positional_groups` dataset since all of them now just had a team and overall score column. Finally, realizing that all of this would mess up the scaling I previously peformed (especially since I only dropped the, on average, lower scoring players), I re-scaled the overall score column, and then I was able to create the regression model for this dataset. Here's what that whole process looked like:

```
# Drop rows where total snaps is less than the mean for that dataset
cb = cb[cb['TOTAL_SNAPS'] > cb['TOTAL_SNAPS'].mean()]
edge = edge[edge['TOTAL_SNAPS'] > edge['TOTAL_SNAPS'].mean()]
s = s[s['TOTAL_SNAPS'] > s['TOTAL_SNAPS'].mean()]
di = di[di['TOTAL_SNAPS'] > di['TOTAL_SNAPS'].mean()]
lb = lb[lb['TOTAL_SNAPS'] > lb['TOTAL_SNAPS'].mean()]

# create singular positional groups dataset
pos_groups_data = cb[['TEAM', 'OVERALL']]
pos_groups_data = pos_groups_data.append(edge[['TEAM', 'OVERALL']], ignore_index = True)
pos_groups_data = pos_groups_data.append(s[['TEAM', 'OVERALL']], ignore_index = True)
pos_groups_data = pos_groups_data.append(di[['TEAM', 'OVERALL']], ignore_index = True)
pos_groups_data = pos_groups_data.append(lb[['TEAM', 'OVERALL']], ignore_index = True)

# Reset the index in case strange indexes could cause issues in the future
pos_groups_data.reset_index(drop = True, inplace = True)
pos_groups_data.head()

pos_groups_data = scale_cols(pos_groups_data, ["OVERALL"])
pos_groups_data = pos_groups_data[["TEAM", "OVERALL_scaled"]]
pos_groups_data.head()
```

The models didn't stop at these, though, as I needed to create some models with logistic regression since some of the columns were binary values (i.e. the interceptions, fumbles, and touchdowns from the play-by-play dataset). For these cases, I couldn't simply use the `lm()` function because this assumes a continuous response variable and a gaussian distribution, and these binary columns were not either of these. Instead of `lm()`, I opted to use the more general version that allowed me to specify that I wanted to do logistic regression: `glm()`. To tell `glm()` that I wanted to perform logistic regression, I had to pass the additional family parameter specifying `binomial(link = "logit")`. The process for these binary columns was as follows:

```
%R int_model = glm(interception ~ defteam, binomial(link = "logit"), data = pbp)
interceptions_model = %R as.data.frame(summary(int_model)$coefficients)
``` 

With that I had all of my models created, but I wasn't quite done yet. Since my models consisted of a categorical independent variable, these were converted to dummy variables for the purposes of regression. Because of this, one of the teams in each model was used as the baseline or intercept, with all other team scores in the model being marked in relation to that baseline/intercept. Thus, in order to get the true scores for each team rather than just their offset from the baseline/intercept team, I had to go through and add the baseline/intercept value for each column to the rest of the rows in the dataset in order to have the scores on the same level as one another. Here's an example of how I took care of that with Python code for one of the datasets, though I did have to apply this same logic to each dataset: 

```
# Grab all rows from the second row on and the two numeric columns and add the value from the first row and numeric 
# columns to each of the other rows, and essentially do this operation 'inplace' by reassigning the values back
yards_by_defteam_model.iloc[1:,1:] = yards_by_defteam_model.iloc[1:,1:] + yards_by_defteam_model.iloc[0, 1:]
```

Once that was all taken care of, I was able to start combining these scores, starting with just getting them into a single dataset. I accomplished this with a few simple `df.merge()`s:
```
coefs = yards_by_defteam_model.merge(interceptions_model, on = 'team')
coefs = coefs.merge(fumbles_model, on = 'team')
coefs = coefs.merge(TDs_model, on = 'team')
coefs = coefs.merge(pos_groups_model, on = 'team')
coefs
```
|team |yards_gained_coef |air_yards_coef |yards_after_catch_coef |interceptions_coef |fumbles_coef |TDs_coef |overall_score_coef|
|-----|------------------|---------------|-----------------------|-------------------|-------------|---------|------------------|
|ARI |0.962836 |-0.206153 |0.026467 |-4.48277 |-4.1352 |-3.55564 |-0.0626037|
|ATL |0.941597 |-0.200649 |-0.00449996 |-4.48443 |-4.31779 |-3.47382 |-0.431962|
|BAL |0.920904 |-0.193628 |-0.0364335 |-4.47491 |-4.07934 |-3.70488 |0.238205|
|BUF |0.936368 |-0.191848 |-0.0220868 |-4.35929 |-4.09977 |-3.44515 |0.127975|
|CAR |0.883663 |-0.218682 |-0.0425162 |-4.44313 |-4.00107 |-3.55194 |-0.107758|
|CHI |0.909072 |-0.181864 |-0.0660687 |-4.41778 |-4.12731 |-3.48424 |1.01037|
|CIN |0.879945 |-0.243177 |-0.0120292 |-4.5219 |-4.35827 |-3.61351 |-0.430532|
|CLE |0.948555 |-0.229398 |0.0432853 |-4.64097 |-4.20108 |-3.53103 |-0.351869|
|DAL |0.931259 |-0.230061 |0.0252596 |-4.70019 |-4.18853 |-3.48412 |0.253901|
|DEN |0.920183 |-0.228842 |0.0120104 |-4.63794 |-4.06333 |-3.55069 |0.16702|
|DET |0.923103 |-0.225549 |0.0115592 |-4.60928 |-4.16361 |-3.47451 |0.118679|
|GB |1.0211 |-0.153685 |0.0171101 |-4.24349 |-4.11419 |-3.47818 |-0.117115|
|HOU |0.959699 |-0.190657 |0.00141954 |-4.54695 |-4.29011 |-3.51871 |0.53005|
|IND |0.912997 |-0.224995 |-0.00193169 |-4.75669 |-4.20032 |-3.53658 |0.226105|
|JAX |0.934259 |-0.201616 |-0.0105754 |-4.72577 |-4.2621 |-3.46514 |0.141497|
|KC |1.0313 |-0.143089 |0.01323 |-4.56753 |-4.21488 |-3.51904 |-0.0838529|
|LA |0.959855 |-0.230975 |0.0582635 |-4.57066 |-4.06104 |-3.48892 |0.22187|
|LAC |0.93116 |-0.202565 |-0.0130441 |-4.55086 |-4.18157 |-3.49577 |-0.157893|
|MIA |0.985397 |-0.179763 |0.0145545 |-4.61121 |-4.22621 |-3.58046 |-0.412654|
|MIN |0.895612 |-0.247252 |0.0105383 |-4.66844 |-4.23322 |-3.60135 |0.24728|
|NE |1.01232 |-0.134598 |-0.0218815 |-4.43595 |-4.0856 |-3.58492 |0.28989|
|NO |1.00106 |-0.201726 |0.0624569 |-4.64266 |-4.1907 |-3.47836 |0.288161|
|NYG |0.994272 |-0.181645 |0.026738 |-4.46413 |-3.99866 |-3.48531 |-0.276424|
|NYJ |0.988548 |-0.154754 |-0.0191688 |-4.62032 |-4.09452 |-3.59743 |0.143248|
|OAK |1.03182 |-0.17456 |0.0571947 |-4.78548 |-4.30609 |-3.40824 |-0.759068|
|PHI |0.985387 |-0.201131 |0.0450565 |-4.42431 |-4.07071 |-3.46956 |0.438634|
|PIT |0.90719 |-0.17813 |-0.0731826 |-4.56047 |-4.13631 |-3.61786 |0.0954372|
|SEA |0.900664 |-0.202853 |-0.0461051 |-4.46969 |-4.08325 |-3.65974 |-0.247267|
|SF |0.911637 |-0.205573 |-0.0302208 |-4.49351 |-4.13098 |-3.58729 |-0.504963|
|TB |0.963882 |-0.180503 |-0.00717164 |-4.45944 |-4.19124 |-3.37829 |-0.727305|
|TEN |0.911121 |-0.197945 |-0.0392834 |-4.60973 |-4.30292 |-3.53101 |0.139264|
|WAS |0.995174 |-0.180708 |0.0264025 |-4.62509 |-3.9891 |-3.50045 |0.180374|

As you can tell from the above table, some of the coefficients were on a completely different scale, namely the ones that game from logistic regression. So, I decided to scale these columns once more to get them all on a level playing field:
```coefs_scaled = scale_cols(coefs, cols = list(coefs.columns)[1:])```

The next thing I needed to take care of was weighting the scaled columns. I was originally planning to weight yards, touchdowns, interceptions, and fumbles according to the amount of points they actually contribute to fantasy football scores; however, this would introduce some issues because touchdowns are by far worth the most points (i.e. 6) while yards are worth relatively few when you boil it down to the yard (i.e. 0.1 points per rushing or receiving yard or 0.04 points per passing yard), yet yards are a much more common means of receiving fantasy points than touchdowns. Another issue was that I have three columns that are related to yards but just one related to touchdowns, thus not weighting yards might bias the model toward teams that are good at stopping yardage. Yet, as mentioned, yards are a more frequent occurance than touchdowns, so honestly this is a good thing as [this CBS Boston article](https://boston.cbslocal.com/2017/09/19/nfl-scoring-touchdown-problem/) explains that there is an average of just under 2.5 touchdowns per team per game. For this reason, I didn't weight the TDs column since TDs are worth a significant number of points but I also didn't weight the yards columns so that this would help to factor in the more likely contribution of yards to fantasy football points.

I still had to consider what to do with interceptions and fumbles. These are worth negative points, so at the very least I had to weight these columns by -1; however, these events, like touchdowns, don't happen terribly often. In fact, according to this [NFL by the Numbers](http://www.sportsonearth.com/article/64441086/nfl-statistical-analysis-average-nfl-game) article, they are essentially just as common as touchdowns, yet they're worth -1/3 times the points compared to touchdowns. For this reason, I weighted the `interceptions_coef_scaled` and `fumbles_coef_scaled` columns by -0.34.

Finally, I had to consider how to weigh the `overall_score_coef_scaled` column. Unfortunately, this column doesn't directly translate into points but rather is an indicator of the strength of a defensive team overall. Since this represents the overall strength of a defense, the higher the number the more difficult, at least theoretically, it would be to score fantasy points against that team, thus I decided to just weight this column by -1. Here's how all of this played out in code:

```
coefs_weighted = coefs_scaled[['team', 'yards_gained_coef_scaled', 'air_yards_coef_scaled', 
                               'yards_after_catch_coef_scaled', 'TDs_coef_scaled']].copy()
coefs_weighted['interceptions_coef_weighted'] = coefs_scaled['interceptions_coef_scaled'] * -0.34
coefs_weighted['fumbles_coef_weighted'] = coefs_scaled['fumbles_coef_scaled'] * -0.34
coefs_weighted['overall_score_coef_weighted'] = coefs_scaled['overall_score_coef_scaled'] * -1
coefs_weighted
```
|team |yards_gained_coef_scaled |air_yards_coef_scaled |yards_after_catch_coef_scaled |TDs_coef_scaled |interceptions_coef_weighted |fumbles_coef_weighted |overall_score_coef_weighted|
|-----|-------------------------|----------------------|------------------------------|----------------|---------------------------|-----------------------------|---------------------------|
|ARI |0.303210 |-0.317428 |0.767608 |-0.414802 |-0.195154 |-0.086068 |0.186097|
|ATL |-0.188828 |-0.116537 |-0.136215 |0.755122 |-0.190362 |0.564677 |1.190459|
|BAL |-0.668235 |0.139684 |-1.068249 |-2.548858 |-0.217927 |-0.285129 |-0.631866|
|BUF |-0.309978 |0.204648 |-0.649515 |1.165091 |-0.552787 |-0.212324 |-0.332127|
|CAR |-1.531017 |-0.774661 |-1.245783 |-0.361889 |-0.309970 |-0.564074 |0.308882|
|CHI |-0.942349 |0.569003 |-1.933202 |0.606131 |-0.383387 |-0.114165 |-2.731544|
|CIN |-1.617149 |-1.668598 |-0.355969 |-1.242252 |-0.081824 |0.708975 |1.186570|
|CLE |-0.027646 |-1.165714 |1.258476 |-0.062885 |0.263024 |0.148755 |0.972669|
|DAL |-0.428349 |-1.189907 |0.732365 |0.607844 |0.434527 |0.104029 |-0.674545|
|DEN |-0.684948 |-1.145443 |0.345668 |-0.344065 |0.254247 |-0.342187 |-0.438299|
|DET |-0.617290 |-1.025259 |0.332499 |0.745335 |0.171236 |0.015189 |-0.306847|
|GB |1.653074 |1.597366 |0.494510 |0.692754 |-0.888193 |-0.160914 |0.334325|
|HOU |0.230543 |0.248120 |0.036555 |0.113237 |-0.009278 |0.466024 |-1.425452|
|IND |-0.851426 |-1.005057 |-0.061256 |-0.142347 |0.598170 |0.146035 |-0.598963|
|JAX |-0.358850 |-0.151822 |-0.313539 |0.879195 |0.508613 |0.366207 |-0.368896|
|KC |1.889308 |1.984080 |0.381264 |0.108536 |0.050308 |0.197937 |0.243878|
|LA |0.234143 |-1.223291 |1.695641 |0.539245 |0.059388 |-0.350371 |-0.587447|
|LAC |-0.430641 |-0.186464 |-0.385589 |0.441271 |0.002053 |0.079193 |0.445209|
|MIA |0.825883 |0.645684 |0.419922 |-0.769728 |0.176825 |0.238291 |1.137956|
|MIN |-1.254181 |-1.817301 |0.302700 |-1.068451 |0.342575 |0.263297 |-0.656543|
|NE |1.449541 |2.293942 |-0.643523 |-0.833526 |-0.330777 |-0.262821 |-0.772406|
|NO |1.188769 |-0.155848 |1.818033 |0.690234 |0.267920 |0.111745 |-0.767707|
|NYG |1.031506 |0.576987 |0.775515 |0.590906 |-0.249141 |-0.572666 |0.767519|
|NYJ |0.898904 |1.558366 |-0.564349 |-1.012400 |0.203215 |-0.231031 |-0.373657|
|OAK |1.901413 |0.835539 |1.664445 |1.692828 |0.681551 |0.523004 |2.079931|
|PHI |0.825662 |-0.134133 |1.310173 |0.816016 |-0.364489 |-0.315884 |-1.176873|
|PIT |-0.985953 |0.705280 |-2.140833 |-1.304584 |0.029877 |-0.082100 |-0.243649|
|SEA |-1.137150 |-0.196986 |-1.350532 |-1.903341 |-0.233041 |-0.271197 |0.688234|
|SF |-0.882931 |-0.296254 |-0.886920 |-0.867455 |-0.164047 |-0.101077 |1.388966|
|TB |0.327445 |0.618662 |-0.214193 |2.121182 |-0.262726 |0.113670 |1.993560|
|TEN |-0.894879 |-0.017858 |-1.151429 |-0.062622 |0.172544 |0.511711 |-0.362823|
|WAS |1.052398 |0.611201 |0.765723 |0.374275 |0.217032 |-0.606732 |-0.474609|

With all of this in place, I could finally amalgamate all of my values. An important note here is that more negative values mean that the particular team has a stronger defense and thus is more difficult:

```
team_scores = pd.DataFrame(coefs_weighted['team'])
team_scores['score'] = coefs_weighted.iloc[:,1:].sum(axis = 1)
team_scores
```
|team |score|
|-----|-----|
|ARI |0.243463|
|ATL |1.878315|
|BAL |-5.280580|
|BUF |-0.686992|
|CAR |-4.478512|
|CHI |-4.929514|
|CIN |-3.070247|
|CLE |1.386677|
|DAL |-0.414036|
|DEN |-2.355028|
|DET |-0.685138|
|GB |3.722924|
|HOU |-0.340251|
|IND |-1.914844|
|JAX |0.560907|
|KC |4.855311|
|LA |0.367308|
|LAC |-0.034969|
|MIA |2.674834|
|MIN |-3.887904|
|NE |0.900430|
|NO |3.153147|
|NYG |2.920627|
|NYJ |0.479049|
|OAK |9.378711|
|PHI |0.960472|
|PIT |-4.021963|
|SEA |-4.404013|
|SF |-1.809719|
|TB |4.697601|
|TEN |-1.805355|
|WAS |1.939287|

Sorting these really tells us a lot more:
|team |score|
|-----|-----|
|BAL |-5.280580|
|CHI |-4.929514|
|CAR |-4.478512|
|SEA |-4.404013|
|PIT |-4.021963|
|MIN |-3.887904|
|CIN |-3.070247|
|DEN |-2.355028|
|IND |-1.914844|
|SF |-1.809719|
|TEN |-1.805355|
|BUF |-0.686992|
|DET |-0.685138|
|DAL |-0.414036|
|HOU |-0.340251|
|LAC |-0.034969|
|ARI |0.243463|
|LA |0.367308|
|NYJ |0.479049|
|JAX |0.560907|
|NE |0.900430|
|PHI |0.960472|
|CLE |1.386677|
|ATL |1.878315|
|WAS |1.939287|
|MIA |2.674834|
|NYG |2.920627|
|NO |3.153147|
|GB |3.722924|
|TB |4.697601|
|KC |4.855311|
|OAK |9.378711|

From the above table, it is clear that the Baltimore Ravens have the strongest defense while the Oakland Raiders have the weakest defense. This tells me that I don't want to pick players from teams that have to play the Ravens a lot, or if one of the players on my fantasy team is playing against the Ravens in any particular week I probably shouldn't play them that week since they are likely to have a tough game. 

In order to get even more out of this, I wanted to determine which teams have the easiest versus most difficult schedules so that I can factor in the relative difficulty of a teams schedule into my decision making for which players to select during my fantasy football drafts. To facilitate this, I obtained a CSV version of this upcoming years NFL schedule and cleaned up inconsistencies between team names from the above table and the schedule dataset. In the schedule dataset, the first column is a team and the remaining columns represent a week and who the team in the team column is playing that week. So, to create a schedule difficulty dataset, I went through and replaced each opposing team value in the schedule dataset with the score for that opposing team from the team_scores dataset:

```
for i in range(len(team_scores)):
    # grab the team name and score for that team
    team = team_scores.iloc[i,0]
    score = team_scores.iloc[i,1]
    # replace all instances of team name except in TEAM column with it's score from the team_scores dataset
    schedule.iloc[:,1:].replace(team, score, inplace = True)
schedule
```
|TEAM |1 |2 |3 |4 |5 |6 |7 |8 |9 |10 |11 |12 |13 |14 |15 |16|
|-----|--|--|--|--|--|--|--|--|--|---|---|---|---|---|---|--|
|ARI |-0.685138 |-5.28058 |-4.47851 |-4.40401 |-3.07025 |1.87832 |2.92063 |3.15315 |-1.80972 |4.6976 |-1.80972 |0 |0.367308 |-4.02196 |1.38668 |-4.40401|
|ATL |-3.8879 |0.960472 |-1.91484 |-1.80536 |-0.340251 |0.243463 |0.367308 |-4.40401 |0 |3.15315 |-4.47851 |4.6976 |3.15315 |-4.47851 |-1.80972 |0.560907|
|BAL |2.67483 |0.243463 |4.85531 |1.38668 |-4.02196 |-3.07025 |-4.40401 |0 |0.90043 |-3.07025 |-0.340251 |0.367308 |-1.80972 |-0.686992 |0.479049 |1.38668|
|BUF |0.479049 |2.92063 |-3.07025 |0.90043 |-1.80536 |0 |2.67483 |0.960472 |1.93929 |1.38668 |2.67483 |-2.35503 |-0.414036 |-5.28058 |-4.02196 |0.90043|
|CAR |0.367308 |4.6976 |0.243463 |-0.340251 |0.560907 |4.6976 |0 |-1.80972 |-1.80536 |3.72292 |1.87832 |3.15315 |1.93929 |1.87832 |-4.40401 |-1.91484|
|CHI |3.72292 |-2.35503 |1.93929 |-3.8879 |9.37871 |0 |3.15315 |-0.0349686 |0.960472 |-0.685138 |0.367308 |2.92063 |-0.685138 |-0.414036 |3.72292 |4.85531|
|CIN |-4.40401 |-1.80972 |-0.686992 |-4.02196 |0.243463 |-5.28058 |0.560907 |0.367308 |0 |-5.28058 |9.37871 |-4.02196 |0.479049 |1.38668 |0.90043 |2.67483|
|CLE |-1.80536 |0.479049 |0.367308 |-5.28058 |-1.80972 |-4.40401 |0 |0.90043 |-2.35503 |-0.686992 |-4.02196 |2.67483 |-4.02196 |-3.07025 |0.243463 |-5.28058|
|DAL |2.92063 |1.93929 |2.67483 |3.15315 |3.72292 |0.479049 |0.960472 |0 |2.92063 |-3.8879 |-0.685138 |0.90043 |-0.686992 |-4.92951 |0.367308 |0.960472|
|DEN |9.37871 |-4.92951 |3.72292 |0.560907 |-0.0349686 |-1.80536 |4.85531 |-1.91484 |1.38668 |0 |-3.8879 |-0.686992 |-0.0349686 |-0.340251 |4.85531 |-0.685138|
|DET |0.243463 |-0.0349686 |0.960472 |4.85531 |0 |3.72292 |-3.8879 |2.92063 |9.37871 |-4.92951 |-0.414036 |1.93929 |-4.92951 |-3.8879 |4.6976 |-2.35503|
|GB |-4.92951 |-3.8879 |-2.35503 |0.960472 |-0.414036 |-0.685138 |9.37871 |4.85531 |-0.0349686 |-4.47851 |0 |-1.80972 |2.92063 |1.93929 |-4.92951 |-3.8879|
|HOU |3.15315 |0.560907 |-0.0349686 |-4.47851 |1.87832 |4.85531 |-1.91484 |9.37871 |0.560907 |0 |-5.28058 |-1.91484 |0.90043 |-2.35503 |-1.80536 |4.6976|
|IND |-0.0349686 |-1.80536 |1.87832 |9.37871 |4.85531 |0 |-0.340251 |-2.35503 |-4.02196 |2.67483 |0.560907 |-0.340251 |-1.80536 |4.6976 |3.15315 |-4.47851|
|JAX |4.85531 |-0.340251 |-1.80536 |-2.35503 |-4.47851 |3.15315 |-3.07025 |0.479049 |-0.340251 |0 |-1.91484 |-1.80536 |4.6976 |-0.0349686 |9.37871 |1.87832|
|KC |0.560907 |9.37871 |-5.28058 |-0.685138 |-1.91484 |-0.340251 |-2.35503 |3.72292 |-3.8879 |-1.80536 |-0.0349686 |0 |9.37871 |0.90043 |-2.35503 |-4.92951|
|LA |-4.47851 |3.15315 |1.38668 |4.6976 |-4.40401 |-1.80972 |1.87832 |-3.07025 |0 |-4.02196 |-4.92951 |-5.28058 |0.243463 |-4.40401 |-0.414036 |-1.80972|
|LAC |-1.91484 |-0.685138 |-0.340251 |2.67483 |-2.35503 |-4.02196 |-1.80536 |-4.92951 |3.72292 |9.37871 |4.85531 |0 |-2.35503 |0.560907 |-3.8879 |9.37871|
|MIA |-5.28058 |0.90043 |-0.414036 |-0.0349686 |0 |1.93929 |-0.686992 |-4.02196 |0.479049 |-1.91484 |-0.686992 |1.38668 |0.960472 |0.479049 |2.92063 |-3.07025|
|MIN |1.87832 |3.72292 |9.37871 |-4.92951 |2.92063 |0.960472 |-0.685138 |1.93929 |4.85531 |-0.414036 |-2.35503 |0 |-4.40401 |-0.685138 |-0.0349686 |3.72292|
|NE |-4.02196 |2.67483 |0.479049 |-0.686992 |1.93929 |2.92063 |0.479049 |1.38668 |-5.28058 |0 |0.960472 |-0.414036 |-0.340251 |4.85531 |-3.07025 |-0.686992|
|NO |-0.340251 |0.367308 |-4.40401 |-0.414036 |4.6976 |0.560907 |-4.92951 |0.243463 |0 |1.87832 |4.6976 |-4.47851 |1.87832 |-1.80972 |-1.91484 |-1.80536|
|NYG |-0.414036 |-0.686992 |4.6976 |1.93929 |-3.8879 |0.90043 |0.243463 |-0.685138 |-0.414036 |0.479049 |0 |-4.92951 |3.72292 |0.960472 |2.67483 |1.93929|
|NYJ |-0.686992 |1.38668 |0.90043 |0 |0.960472 |-0.414036 |0.90043 |0.560907 |2.67483 |2.92063 |1.93929 |9.37871 |-3.07025 |2.67483 |-5.28058 |-4.02196|
|OAK |-2.35503 |4.85531 |-3.8879 |-1.91484 |-4.92951 |0 |3.72292 |-0.340251 |-0.685138 |-0.0349686 |-3.07025 |0.479049 |4.85531 |-1.80536 |0.560907 |-0.0349686|
|PHI |1.93929 |1.87832 |-0.685138 |3.72292 |0.479049 |-3.8879 |-0.414036 |-0.686992 |-4.92951 |0 |0.90043 |-4.40401 |2.67483 |2.92063 |1.93929 |-0.414036|
|PIT |0.90043 |-4.40401 |-1.80972 |-3.07025 |-5.28058 |-0.0349686 |0 |2.67483 |-1.91484 |0.367308 |1.38668 |-3.07025 |1.38668 |0.243463 |-0.686992 |0.479049|
|SF |4.6976 |-3.07025 |-4.02196 |0 |1.38668 |0.367308 |1.93929 |-4.47851 |0.243463 |-4.40401 |0.243463 |3.72292 |-5.28058 |3.15315 |1.87832 |0.367308|
|SEA |-3.07025 |-4.02196 |3.15315 |0.243463 |0.367308 |1.38668 |-5.28058 |1.87832 |4.6976 |-1.80972 |0 |0.960472 |-3.8879 |0.367308 |-4.47851 |0.243463|
|TB |-1.80972 |-4.47851 |2.92063 |0.367308 |3.15315 |-4.47851 |0 |-1.80536 |-4.40401 |0.243463 |3.15315 |1.87832 |0.560907 |-1.91484 |-0.685138 |-0.340251|
|TEN |1.38668 |-1.91484 |0.560907 |1.87832 |-0.686992 |-2.35503 |-0.0349686 |4.6976 |-4.47851 |4.85531 |0 |0.560907 |-1.91484 |9.37871 |-0.340251 |3.15315|
|WAS |0.960472 |-0.414036 |-4.92951 |2.92063 |0.90043 |2.67483 |-1.80972 |-3.8879 |-0.686992 |0 |0.479049 |-0.685138 |-4.47851 |3.72292 |0.960472 |2.92063|

With this dataset made, I then created a Heatmap using `seaborn` in order to get a visual representation of the table above:
```
import seaborn as sns
import matplotlib.pyplot as plt

for col in list(schedule.columns):
    schedule[col] = pd.to_numeric(schedule[col])

# set figure size to make visualization more clear
plt.figure(figsize = (15, 20))
# Create heatmap, setting the color mapping to the inverse of coolwarm (signified by _r) 
# because I want red to indicate a difficult game and blue to indicate an easy game
sns.heatmap(schedule, cmap = 'coolwarm_r')
```
![]()

Finally, in order to easily understand which teams have the most difficult versus easiest season overall, I added up the value from each week for each team in order to give them an overall season score:
```
schedule['season_score'] = schedule.sum(axis = 1)
season_scores = pd.DataFrame(schedule['season_score'])
season_scores.sort_values('season_score')
```
|TEAM |season_score|
|-----|------------|
|CLE |-28.071355|
|LA |-23.263111|
|ARI |-15.560227|
|PIT |-12.833171|
|ATL |-9.983064|
|CIN |-9.514430|
|SEA |-9.251169|
|TB |-7.639429|
|GB |-7.357830|
|MIA |-7.045031|
|NO |-5.772732|
|BAL |-5.109681|
|OAK |-4.584716|
|SF |-3.255820|
|BUF |-2.110569|
|WAS |-1.352382|
|KC |0.353072|
|PHI |1.033120|
|NE |1.194245|
|NYG |6.539728|
|HOU |8.201199|
|LAC |8.276373|
|DET |8.279527|
|JAX |8.297322|
|DEN |10.439906|
|DAL |10.809628|
|NYJ |10.823392|
|IND |12.017142|
|CAR |12.864688|
|TEN |14.746138|
|MIN |15.870735|
|CHI |22.958498|

And with all of that I was done with everything I wanted to accomplish!!!

## Conclusions
According to the model, the Cleveland Browns are going to have the toughest go of it this year in terms of schedule difficulty, so I should make sure to avoid drafting players from the Browns in my fantasy drafts. On the other hand, the following are the top 5 teams in terms of ease of schedule, thus I should prioritize drafting players from these teams: 
1. Chicago Bears
2. Minnesota Vikings
3. Tennessee Titans
4. Carolina Panthers
5. Indianapolis Colts

With all of that I have all the information I was hoping to obtain. I have ample ideas for ways to improve my model going forward, such as by separating out running and passing defensive strength and producing a strength of schedule in those regards as well so that I know which teams to pick running backs from as well as wide receivers, and quarterbacks from, respectively. I'd also like to create models based on offensive productivity so that I can then use that in my decisions for which teams to draft from because as it stands, sure the Vikings are supposed to be second and the Titans third in my defensive strength of schedule model, but the difference between these teams is pretty minimal, and what if the Titans have a better overall offense? I'm currently missing out on insights like these, so I plan to make these kinds of improvements in future iterations through my model. 
# TODO 
- add links for notebooks when uploaded
- add link for heatmap visualization

