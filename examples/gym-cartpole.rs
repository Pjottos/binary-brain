extern crate gym;
extern crate binary_brain;

use binary_brain::BinaryBrain;

const INPUT_COUNT: usize = 4;
const OUTPUT_COUNT: usize = 1;
const TOTAL_COUNT: usize = 64;

fn main() {
    let gym = gym::GymClient::default();
    let env = gym.make("CartPole-v1");
    let mut brain = BinaryBrain::new(INPUT_COUNT, OUTPUT_COUNT, TOTAL_COUNT).unwrap();

    
    let mut input = map_observation(env.reset().expect("unable to reset"));
    let mut output = vec![];

    loop {
        println!("feeding input: {:?}", input);
        println!("BRAIN: {:?}", brain);
        brain.cycle(&input, &mut output).unwrap();
        println!("got output: {:?}", output);
        let action = if output[0] {gym::SpaceData::DISCRETE(0)} else {gym::SpaceData::DISCRETE(1)};
        let state = env.step(&action).unwrap();
        input = map_observation(state.observation);
        env.render();
        if state.is_done {
            break;
        }
    }

	env.close();
}

fn map_observation(observation: gym::SpaceData) -> [u8; INPUT_COUNT] {
    let vec = observation.get_box().unwrap();
    
    [
        // cart position (negative is left, positive is right)
        // the allowed range is [-4.8, 4.8] but since the cart is almost never
        // out that far, we prefer better resolution for the lower (absolute) numbers
        (((vec[0] + 2.4) / 4.8) * 255.0) as u8,

        // cart velocity (- left, + right)
        // theoretically this has infinite range but practically it will
        // never go outside this range, so long as the environment is not forced to continue
        (((vec[1] + 3.0) / 6.0) * 255.0) as u8,
        
        // pole angle in radians (- left, + right)
        // this is the allowed range, if the pole tips outside it the env terminates
        (((vec[2] + 0.418) / 0.836) * 255.0) as u8,
        
        // pole angular velocity
        // again, technically infinite
        (((vec[3] + 4.5) / 9.0) * 255.0) as u8,
    ]
}