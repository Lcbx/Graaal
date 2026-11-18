from scenes.postprocessing_example import update_acceleration_boost


def test_acceleration_boost_tracks_changes_and_decays():
	boost = update_acceleration_boost(0, 0, 2, 0.1, 1.0)
	assert boost > 0
	assert update_acceleration_boost(0, 0, -2, 0.1, 1.0) == boost
	assert update_acceleration_boost(boost, 2, 2, 1.0, 1.0) < boost
	assert update_acceleration_boost(0, 10, 8, 0.1, 1.0) == 0
	assert update_acceleration_boost(0, -10, -8, 0.1, 1.0) == 0
	assert update_acceleration_boost(boost, 10, 8, 0.1, 1.0) < boost
	assert update_acceleration_boost(0, 1, -2, 0.1, 1.0) == boost
	assert update_acceleration_boost(boost, 2, 2, 0, 1.0) == boost
	long_step = update_acceleration_boost(0, 0, 20, 1.0, 1.0)
	short_steps = 0
	for step in range(10):
		short_steps = update_acceleration_boost(short_steps, step*2, (step+1)*2, 0.1, 1.0)
	assert abs(long_step-short_steps) < 1e-10
