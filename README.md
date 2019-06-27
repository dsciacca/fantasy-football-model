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

## Data Cleaning
Data from [PFF](https://www.profootballfocus.com/) came very clean from the get-go and didn't require any further cleaning for my initial needs. `nflscrapR` data, however, did take some considerable cleaning. My entire process, which is thoroughly explained, can be seen in the [Fantasy Football Regression Model - nflscrapR data cleaning]() Jupyter notebook, and the highlights are below:
 
### Missing/Null Value Handling
To start, since this dataset contains so many potential features, with some datasets containing over 250 columns, I decided to begin by removing any column that was composed of 80% or greater null values straight off the bat, which managed to get rid of over 80 columns from some of the datasets:  

```data.dropna(axis = 1, thresh = int(0.2 * data.shape[0]), inplace = True)```

From there I took a look at all of the remaining column names in each dataset and determined which ones I knew would not ultimately be part of my model, and I dropped these. This process reduced my number of columns down to closer to 100 in the wider play-by-play datasets. After that I began to dig further into the data. I checked the number of unique team names in each of the datasets to ensure that they aligned with the 32 in the NFL, and I realized that this ligned up well for the pre and regular season datasets, but not for the post season datasets due to not every team being present in the post season. For this reason, I chose not to continue with using the post season play-by-play or games datasets that I obtained from `nflsrcapR`.


# TODO 
- add links for notebooks when uploaded
