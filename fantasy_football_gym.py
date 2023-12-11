import gym
from gym import spaces
import numpy as np
import pandas as pd

df = pd.read_csv('./data/adp_and_scoring.csv')

def action_to_position(action):
    """
    Converts action int to the position str
    """
    if action == 0:
        return 'QB'
    if action == 1:
        return 'RB'
    if action == 2:
        return 'WR'
    if action == 3:
        return 'TE'
    
    raise ValueError('Invalid action')


class SmartAutoDraftAgent():
    def __init__(self):
        """
        Picks best available until reaching position thresholds, following a smart autodraft logic
        """
        pass
        
    def argmin_for_pickable(self, best_available, pickable):
        """
        Picks best available starters, then best available first tier bench
        """
        if (np.sum(pickable) == 0) or (np.sum(pickable) == 4):
            return np.argmin(best_available)

        args = np.argsort(best_available)

        for arg in args:
            if pickable[arg]:
                return arg
        
        raise ValueError('No position was picked')
    
    def make_selection(self, observation):
        num_qbs = observation[4]
        num_rbs = observation[5]
        num_wrs = observation[6]
        num_tes = observation[7]

        # first wave
        can_pick_qb_wave_1 = (num_qbs < 1)
        can_pick_rb_wave_1 = (num_rbs < 2) or (num_rbs <= 2 and num_wrs <= 2 and num_tes <= 1)
        can_pick_wr_wave_1 = (num_wrs < 2) or (num_rbs <= 2 and num_wrs <= 2 and num_tes <= 1)
        can_pick_te_wave_1 = (num_tes < 1) or (num_rbs <= 2 and num_wrs <= 2 and num_tes <= 1)

        if (can_pick_qb_wave_1 + can_pick_rb_wave_1 + can_pick_wr_wave_1 + can_pick_te_wave_1):
            return self.argmin_for_pickable(
                observation[:4], 
                [can_pick_qb_wave_1, can_pick_rb_wave_1, can_pick_wr_wave_1, can_pick_te_wave_1]
            )
        
        # second wave
        can_pick_qb_wave_2 = (num_qbs < 3)
        can_pick_rb_wave_2 = (num_rbs < 5)
        can_pick_wr_wave_2 = (num_wrs < 5)
        can_pick_te_wave_2 = (num_tes < 3)
        
        return self.argmin_for_pickable(
            observation[:4], 
            [can_pick_qb_wave_2, can_pick_rb_wave_2, can_pick_wr_wave_2, can_pick_te_wave_2]
        )


class FantasyFootball(gym.Env):
    def __init__(self, df_all=df, num_teams=10, num_rounds=16, num_players=400):
        super(FantasyFootball, self).__init__()

        self.df_all = df_all
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.num_players = num_players

        # set the agent draft order
        self.agent_team_index = np.random.randint(num_teams)

        # define action space and stuff
        # self.observation_space = spaces.Tuple((
        #     spaces.Discrete(num_players),  # best QB available
        #     spaces.Discrete(num_players),  # best RB available
        #     spaces.Discrete(num_players),  # best WR available
        #     spaces.Discrete(num_players),  # best TE available
        #     spaces.Discrete(num_rounds),   # Amount of QBs on roster
        #     spaces.Discrete(num_rounds),   # Amount of RBs on roster
        #     spaces.Discrete(num_rounds),   # Amount of WRs on roster
        #     spaces.Discrete(num_rounds),   # Amount of TEs on roster
        # ))

        self.observation_space = spaces.Box(low=np.array([0]*8), high=np.array([num_players]*4 + [num_rounds]*4), dtype=np.int16)
        self.action_space = spaces.Discrete(4)

    
    def reset(self):
        # recreate draft board

        data = [
            self.df_all.loc[
                (self.df_all['Year'] == np.random.choice(self.df_all['Year'].unique())) & 
                (self.df_all['ADP'] == round(np.random.normal(adp, 1 + adp/10))), 
                :
            ] for adp in range(1, self.num_players + 1)
        ]

        self.df_draft_board = pd.concat(data, ignore_index=True)

        # recompute adps based on draft board
        self.df_draft_board['ADP'] = self.df_draft_board.index + 1

        # recompile draft order
        self.teams = [pd.DataFrame(columns=self.df_draft_board.columns) for _ in range(self.num_teams)]

        # start the draft
        self.current_draft_round = 0
        self.current_pick_number = 0
        self.team_index = self.get_team_index()

        # return observation
        self.observation = self.get_observation()
        
        self.observation, reward, done, info = self.advance_environment()

        return self.observation


    def step(self, action):
        done = False
        reward = 0
        
        # make selection and update draft board
        self.draft_player(action)

        self.current_pick_number += 1

        if self.current_pick_number >= self.num_teams:
            self.current_pick_number = 0
            self.current_draft_round += 1

        if self.current_draft_round >= self.num_rounds:
            return None, self.season_placements(), True, {}
        
        self.observation = self.get_observation()

        return self.advance_environment()

        # return self.observation, reward, done, {}

    
    def advance_environment(self):
        """
        get the environment to a point where either the agent is ready to draft again or the draft is over
        """
        # while it is not agent's pick
        while (self.agent_team_index != self.team_index) and (self.agent_team_index - self.num_teams != self.team_index):
            # make pick accoring to computer strategy
            action = SmartAutoDraftAgent().make_selection(self.observation)
            self.draft_player(action)

            # advance to next pick
            self.current_pick_number += 1

            if self.current_pick_number >= self.num_teams:
                self.current_pick_number = 0
                self.current_draft_round += 1

            # if draft is over
            if self.current_draft_round >= self.num_rounds:
                return self.observation, self.season_placements(), True, {}
        
            self.observation = self.get_observation()

        return self.observation, 0, False, {}


    def get_observation(self):
        """
        get observations with respect to the team that is about to pick
        """
        observations = []        
        self.team_index = self.get_team_index()

        # best player available
        for position in ('QB', 'RB', 'WR', 'TE'):
            try:
                observations.append(self.df_draft_board.loc[self.df_draft_board['Position'] == position, 'ADP'].iloc[0])
            except IndexError:
                observations.append(self.num_players + 1)
        
        # current roster makeup
        for position in ('QB', 'RB', 'WR', 'TE'):
            observations.append(self.teams[self.team_index].loc[self.teams[self.team_index]['Position'] == position, :].shape[0])
        
        return observations
    
    def get_team_index(self):
        if self.current_draft_round % 2 == 0: 
            return self.current_pick_number
        else:
            return -self.current_pick_number - 1
    
    
    def draft_player(self, action: int):
        """
        select a player to be drafted, placing them on the team, and removing them from the draft board
        """
        position = action_to_position(action)
        df_pos = self.df_draft_board.loc[self.df_draft_board['Position'] == position, :]

        # find player and update board
        if df_pos.shape[0]:
            df_drafted_player = df_pos.iloc[:1]
            self.df_draft_board = self.df_draft_board.loc[self.df_draft_board.index != df_drafted_player.index[0], :]
        else:
            # in case there are no players of a given position left to draft
            df_drafted_player = pd.DataFrame(
                [[0, self.num_players + 1, 'PLACEHOLDER', position] + [0 for _ in range(1, 17)]], 
                columns=self.df_draft_board.columns
            )

        # put player on team
        self.teams[self.team_index] = pd.concat([self.teams[self.team_index], df_drafted_player], ignore_index=True)

        return
        
    
    def score_week(self, week):
        """
        get fantasy points scored for a given week
        """

        scores = []
        for df_team in self.teams:
            
            qb_scores = np.array(df_team.loc[df_team['Position'] == 'QB', f'{week}'].sort_values(ascending=False).iloc[:1])
            qb_scores = np.concatenate((qb_scores, np.zeros(shape=(1 - qb_scores.shape[0]))))

            rb_scores = np.array(df_team.loc[df_team['Position'] == 'RB', f'{week}'].sort_values(ascending=False).iloc[:3])
            rb_scores = np.concatenate((rb_scores, np.zeros(shape=(3 - rb_scores.shape[0]))))

            wr_scores = np.array(df_team.loc[df_team['Position'] == 'WR', f'{week}'].sort_values(ascending=False).iloc[:3])
            wr_scores = np.concatenate((wr_scores, np.zeros(shape=(3 - wr_scores.shape[0]))))

            te_scores = np.array(df_team.loc[df_team['Position'] == 'TE', f'{week}'].sort_values(ascending=False).iloc[:2])
            te_scores = np.concatenate((te_scores, np.zeros(shape=(2 - te_scores.shape[0]))))

            score = (
                qb_scores.sum() + 
                rb_scores[:2].sum() + 
                wr_scores[:2].sum() + 
                te_scores[:1].sum() + 
                np.max((rb_scores[2], wr_scores[2], te_scores[1]))
            )

            scores.append(score)
            
        return scores
    
    def season_placements(self):
        season_scores = np.array([self.score_week(week) for week in range(1, 17)])
        season_scores_comparison = np.array([
            [(row < value).sum()/(season_scores.shape[1] - 1) for value in row] 
            for row in season_scores
        ])

        return ((season_scores_comparison.mean(axis=0) - 0.5)*2)[self.agent_team_index]
    

if __name__ == "__main__":
    env = FantasyFootball()
    print(env)