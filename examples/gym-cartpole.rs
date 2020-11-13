extern crate gym;
extern crate binary_brain;

use binary_brain::BinaryBrain;
use binary_brain::train;

// model params
const INPUT_COUNT: usize = 4;
const OUTPUT_COUNT: usize = 1;
const TOTAL_COUNT: usize = 64;

// genetic trainer params
const POPULATION_SIZE: usize = 256;
const TOURNAMENT_SIZE: usize = 8;
const MUTATION_CHANCE: f64 = 0.015;
const CROSSOVER_SWITCH_CHANCE: f64 = 0.4;

const MAX_GENERATIONS: usize = 32;

fn main() {
    let model = BinaryBrain::new(INPUT_COUNT, OUTPUT_COUNT, TOTAL_COUNT).unwrap();
    let mut trainer = train::Genetic::new(model, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION_CHANCE, CROSSOVER_SWITCH_CHANCE).unwrap();
    
    println!("starting training...");

    for i in 0..MAX_GENERATIONS {
        // you could also use trainer.evaluate_parallel if you have a more expensive simulation
        // since the cartpole environment is very cheap to simulate, it's faster to run single-threaded
        let top_fitness = trainer.evaluate(|brain| {
            let gym = gym::GymClient::default();
            let env = gym.make("CartPole-v1");
            let mut input = map_observation(env.reset().unwrap());
            let mut output = Vec::with_capacity(OUTPUT_COUNT);
            let mut fitness = 0.0;

            loop {
                brain.cycle(&input, &mut output).unwrap();
                let action = if output[0] {gym::SpaceData::DISCRETE(0)} else {gym::SpaceData::DISCRETE(1)};
                let state = env.step(&action).unwrap();
                input = map_observation(state.observation);
                if state.is_done {
                    break;
                }
                fitness += state.reward;
            }

            env.close();

            fitness
        });

        println!("generation {} finished, top fitness: {}", i, top_fitness);

        // the enviroment reports it is done above this fitness
        // so training any further would not reinforce correct behaviour anymore
        if top_fitness < 499.0 {
            trainer.breed();
        } else {
            break;
        }
    }

    println!("training finished, rendering fittest individual...");

    let mut fittest = trainer.clone_fittest().0;

    let gym = gym::GymClient::default();
    let env = gym.make("CartPole-v1");
    let mut input = map_observation(env.reset().unwrap());
    let mut output = vec![];
    
    loop {
        fittest.cycle(&input, &mut output).unwrap();
        let action = if output[0] {gym::SpaceData::DISCRETE(0)} else {gym::SpaceData::DISCRETE(1)};
        let state = env.step(&action).unwrap();
        env.render();
        std::thread::sleep(std::time::Duration::from_millis(20));
        input = map_observation(state.observation);
        if state.is_done {
            break;
        }
    }

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