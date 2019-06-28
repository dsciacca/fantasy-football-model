# Fantasy Football Regression Model - Using Team and Player Stats to Improve Fantasy Team Performance

## Introduction
In Fantasy Football, oftentimes, when deciding on whom to draft for a fantasy football team, users focus on their own team and player preference (i.e. their favorite teams and favorite players) and occasionally individual players past year’s performance. The problem here is that only the players past year’s performance is quantifiable, and emotional decisions like team and player preference do not have a quantifiable reasoning behind them, which can easily lead to poor overall fantasy team performance. In addition to this, users then miss out on otherwise extremely valuable insights, such as the past performance of the actual team that the individual players belong to (after all, football is a team sport), how various off-season trades and NFL draft picks will influence the performance of that team in the next year, and the expected difficulty of a teams schedule in the coming season (i.e. a receiver on a team that is playing a lot of other teams with a great passing defenses will have a more difficult time scoring fantasy points than a receiver on a team that is more often playing teams with poor passing defenses). This project addresses some of these issue by using both player and team stats and the upcoming season’s schedule to determine which teams are the best teams to draft players from.

## The Data
In this project, I make use of the freely available NFL team, game, and player [datasets](https://github.com/ryurko/nflscrapR-data) from the [nflscrapR package](https://github.com/maksimhorowitz/nflscrapR) - which contains hundreds of thousands of rows and hundreds of columns. From this package, I specifically made use of the following datasets:
- All pre season, regular season, and post season [play-by-play datasets](https://github.com/ryurko/nflscrapR-data/tree/master/play_by_play_data) from 2009 to 2018
- All pre season, regular season, and post season [game datasets](https://github.com/ryurko/nflscrapR-data/tree/master/games_data) from 2009 to 2018

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

With this taken care of, I then began down the path of exploring my data. 

# TODO 
- add links for notebooks when uploaded
