environment=Humanoid  # needs OpenAI-Gym MuJoCo installed
seed=0
python3 -um hyperdopamine.interfaces.train \
  --base_dir=./logs/$environment/bdqn/$seed \
  --gin_files='hyperdopamine/agents/bdqn/configs/bdqn.gin' \
  --gin_bindings="BDQNAgent.observation_shape=%environment_metadata.${environment^^}_OBSERVATION_SHAPE" \
  --gin_bindings="create_discretised_environment.environment_name='$environment'" \
  --gin_bindings="create_discretised_environment.version=%environment_metadata.${environment^^}_ENV_VER" \
  --gin_bindings="create_discretised_environment.environment_seed=$seed" \
  --gin_bindings="Runner.max_steps_per_episode=%environment_metadata.${environment^^}_TIMELIMIT" \
  --gin_bindings="Runner.agent_seed=$seed" \
  --gin_bindings="WrappedPrioritizedReplayBuffer.action_shape=%environment_metadata.${environment^^}_ACTION_SHAPE" \
  --gin_bindings="WrappedPrioritizedReplayBuffer.reward_shape=1" \
  --schedule=continuous_train_and_eval