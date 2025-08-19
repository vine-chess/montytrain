pub mod data;
pub mod inputs;
pub mod model;

use bullet_core::{
    device::Device,
    optimiser::{
        adam::{AdamW, AdamWParams},
        Optimiser,
    },
    trainer::{
        schedule::{TrainingSchedule, TrainingSteps},
        Trainer,
    },
};
use bullet_cuda_backend::CudaDevice;

use crate::data::MontyDataLoader;

fn main() {
    let hl = 256;
    let dataloader = MontyDataLoader::new("data/output_08_09_higher_root_pst_2_3.bin", 8192, 16, 16);

    let device = CudaDevice::new(0).unwrap();

    let (graph, node) = model::make(device, hl);

    let params = AdamWParams { decay: 0.01, beta1: 0.9, beta2: 0.999, min_weight: -0.99, max_weight: 0.99 };
    let optimiser = Optimiser::<_, AdamW<_>>::new(graph, params).unwrap();

    let mut trainer = Trainer { optimiser, state: () };

    let save_rate = 10;
    let end_superbatch = 50;
    let initial_lr = 0.001;
    let final_lr = 0.00001;

    let steps = TrainingSteps { batch_size: 16384, batches_per_superbatch: 6104, start_superbatch: 1, end_superbatch };

    let schedule = TrainingSchedule {
        steps,
        log_rate: 64,
        lr_schedule: Box::new(|_, sb| {
            if sb >= end_superbatch {
                return final_lr;
            }

            let lambda = sb as f32 / end_superbatch as f32;
            initial_lr + (final_lr - initial_lr) * lambda
        }),
        // lr_schedule: Box::new(|_, sb| {
        //     if sb >= end_superbatch {
        //         return final_lr;
        //     }
        //
        //     let lambda = sb as f32 / end_superbatch as f32;
        //     initial_lr * (final_lr / initial_lr).powf(lambda)
        // }),
    };

    trainer
        .train_custom(
            schedule,
            dataloader,
            |_, _, _, _| {},
            |trainer, superbatch| {
                if superbatch % save_rate == 0 || superbatch == steps.end_superbatch {
                    println!("Saving Checkpoint");
                    let dir = format!("checkpoints/policy-{superbatch}");
                    let _ = std::fs::create_dir(&dir);
                    trainer.optimiser.write_to_checkpoint(&dir).unwrap();
                    model::save_quantised(&trainer.optimiser.graph, &format!("{dir}/quantised.bin")).unwrap();
                }
            },
        )
        .unwrap();
    // _ = trainer.optimiser.load_from_checkpoint("checkpoints/policy-50");

    model::eval(&mut trainer.optimiser.graph, node, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    model::eval(&mut trainer.optimiser.graph, node, "rnbqkbnr/ppppp2p/5p2/6p1/4PP2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3");
    model::eval(&mut trainer.optimiser.graph, node, "rnbqk2r/pppp1ppp/3bpn2/8/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4");
    model::eval(&mut trainer.optimiser.graph, node, "rnbqk3/ppppp1Q1/5p2/6p1/8/8/PPPPPPPP/RNB1KBNR w KQq - 0 1");
}
