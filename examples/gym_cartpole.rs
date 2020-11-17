extern crate gym;
extern crate binary_brain;

use binary_brain::{train, BinaryBrain, Activation};
use std::io;

// model params
const INPUT_COUNT: usize = 4;
const OUTPUT_COUNT: usize = 1;
const TOTAL_COUNT: usize = 64;

const CYCLE_COUNT: usize = 1;

// genetic trainer params
const POPULATION_SIZE: usize = 256;
const TOURNAMENT_SIZE: usize = 16;
const MUTATION: u8 = 2; // mutation per parameter: p = 2/256

const MAX_GENERATIONS: usize = 64;

fn main() {
    println!("pretrained brain to load (leave empty to train):");
    let mut response = String::default();
    io::stdin().read_line(&mut response).unwrap();
    response.pop();
    
    let mut model;
    let is_trained;
    if response.is_empty() {
        is_trained = true;
        let initial = BinaryBrain::new(INPUT_COUNT, OUTPUT_COUNT, TOTAL_COUNT).unwrap();
        let mut trainer = train::Genetic::new(initial, POPULATION_SIZE, TOURNAMENT_SIZE, MUTATION).unwrap();
        
        println!("starting training...");
        let gym = gym::GymClient::default();
        let env = gym.make("CartPole-v1");
        
        for i in 0..MAX_GENERATIONS {
            let top_fitness = trainer.evaluate(|brain| {
                let mut input = map_observation(env.reset().unwrap());
                let mut output = Vec::with_capacity(OUTPUT_COUNT);
                let mut fitness = 0.0;
    
                loop {
                    for _ in 0..CYCLE_COUNT {
                        brain.cycle(&input, &mut output).unwrap();
                    }
                    let action = if output[0].0 {gym::SpaceData::DISCRETE(0)} else {gym::SpaceData::DISCRETE(1)};
                    let state = env.step(&action).unwrap();
                    input = map_observation(state.observation);
                    fitness += state.reward;
                    if state.is_done {
                        break;
                    }
                }
    
                fitness
            });
    
            println!("generation {} finished, top fitness: {}", i, top_fitness);
    
            // the enviroment reports it is done above this fitness
            // so training any further would not reinforce correct behaviour anymore
            if top_fitness < 500.0 {
                trainer.breed();
            } else {
                println!("achieved maximum fitness");
                break;
            }
        }
        
        println!("training stopped, using fittest brain of last generation");
        env.close();
        model = trainer.clone_fittest().0;
    } else {
        is_trained = false;
        let target = "data/pretrained/".to_owned() + &response;
        println!("loading brain from \"{}\"...", target);

        model = BinaryBrain::from_file(target).unwrap();
    }

    println!("evaluating brain...");

    let gym = gym::GymClient::default();
    let env = gym.make("CartPole-v1");
    let mut input = map_observation(env.reset().unwrap());
    let mut output = vec![];
    let mut fitness = 0.0; 
    
    loop {
        model.cycle(&input, &mut output).unwrap();
        let action = if output[0].0 {gym::SpaceData::DISCRETE(0)} else {gym::SpaceData::DISCRETE(1)};
        let state = env.step(&action).unwrap();
        env.render();
        std::thread::sleep(std::time::Duration::from_millis(20));
        input = map_observation(state.observation);
        fitness += state.reward;
        if state.is_done {
            break;
        }
    }

    env.close();

    println!("evaluation finished, fitness: {}", fitness);

    if is_trained {
        println!("filename to save brain in (leave empty to discard this brain):");
        response.clear();
        io::stdin().read_line(&mut response).unwrap();
        response.pop();
    
        if !response.is_empty() {
            let target = "data/pretrained/".to_owned() + &response;
            println!("saving brain to \"{}\"...", target);
            model.write_to_file(target).unwrap();
        }
    }
}

fn map_observation(observation: gym::SpaceData) -> [Activation; INPUT_COUNT] {
    let vec = observation.get_box().unwrap();

    [
        // cart position (negative is left, positive is right)
        // the allowed range is [-4.8, 4.8] but since the cart is almost never
        // out that far, we prefer better resolution near the center
        Activation(((vec[0] / 2.4) * 255.0) as i8),

        // cart velocity (- left, + right)
        // theoretically this has infinite range but practically it will
        // never go outside this range, so long as the environment is not forced to continue
        Activation(((vec[1] / 3.0) * 255.0) as i8),
        
        // pole angle in radians (- left, + right)
        // this is the allowed range, if the pole tips outside it the env terminates
        Activation(((vec[2] / 0.418) * 255.0) as i8),
        
        // pole angular velocity
        // again, technically infinite
        Activation(((vec[3] / 4.5) * 255.0) as i8),
    ]
}