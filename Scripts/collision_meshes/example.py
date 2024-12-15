import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger
from rlgym.utils.reward_functions.common_rewards import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData 

class InAirReward(RewardFunction): # We extend the class "RewardFunction"
    # Empty default constructor (required)
    def __init__(self):
        super().__init__()

    # Called when the game resets (i.e. after a goal is scored)
    def reset(self, initial_state: GameState):
        pass # Don't do anything when the game resets

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        
        # "player" is the current player we are getting the reward of
        # "state" is the current state of the game (ball, all players, etc.)
        # "previous_action" is the previous inputs of the player (throttle, steer, jump, boost, etc.) as an array
        
        if not player.on_ground:
            # We are in the air! Return full reward
            return 1
        else:
            # We are on ground, don't give any reward
            return 0

from rlgym_sim.utils.common_values import CAR_MAX_SPEED

class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0

class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.orange_score]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, \
        EventReward, FaceBallReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.state_setters.random_state import RandomState
    from lookupaction import LookupAction

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10 # maybe more? 
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]


    # 1st set of rewards
    # rewards_to_combine = (
    #     (EventReward(touch=1)),
    #     (VelocityPlayerToBallReward()),
    #     (FaceBallReward()),
    #     (InAirReward()))
    # reward_weights = (1.0, 0.1, 0.02, 0.01)

    # 2nd set of rewards
    rewards_to_combine = (
        (EventReward(touch=1)),
        (SpeedTowardBallReward()),
        (VelocityBallToGoalReward()),
        (FaceBallReward()),
        (InAirReward()))
    reward_weights = (0.01, 0.1, 0.3, 0.02, 0.01)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    obs_builder = DefaultObs(
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)
    
    state_setter = RandomState(True, True, False)

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         state_setter=state_setter,
                         action_parser=action_parser)

    # Uncomment for visualization
    # import rocketsimvis_rlgym_sim_client as rsv
    # type(env).render = lambda self: rsv.send_state_to_rocketsimvis(self._prev_state)
    
    return env            

if __name__ == "__main__":
    from rlgym_ppo import Learner
    import os
    
    metrics_logger = ExampleLogger()

    # 8 = around 7000, 4750
    # 32 = around 17000, 7000
    n_proc = 24
    
    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    latest_checkpoint_dir = "data/checkpoints/" + str(max(os.listdir("data/checkpoints/"), key=lambda d: int(d[14:])))
    latest_checkpoint = max(os.listdir(latest_checkpoint_dir), key=lambda d: int(d))
    checkpoint_load_folder = latest_checkpoint_dir + "/" + latest_checkpoint
    
    print("Loading from checkpoint: ", checkpoint_load_folder)
    
    learner = Learner(build_rocketsim_env,
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      policy_layer_sizes=[2048, 2048, 1024, 1024],
                      critic_layer_sizes=[2048, 2048, 1024, 1024],
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      ppo_ent_coef=0.01,
                      ppo_epochs=2,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=100_000,
                      timestep_limit=10e15,
                      checkpoint_load_folder=checkpoint_load_folder,
                    #   render=True,
                      log_to_wandb=True) #checkpoint_load_folder=None
    learner.learn()