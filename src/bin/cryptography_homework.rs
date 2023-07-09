use discrete_logarithm_lib::{is_prime, FpUnitsDiscLogSolver};

fn main() {
    println!("593 is prime: {}", is_prime(593));
    let solver = FpUnitsDiscLogSolver::new(593);
    let base = 156;
    let x = solver
        .shanks_algorithm(base, 116)
        .expect("Should have a logarithm");
    println!("log of {} base {} modulo 593 is {}", 116, base, x);
    println!("{}^{} = {}", base, x, solver.fast_power(base, x));

    println!("3571 is prime: {}", is_prime(3571));
    let solver = FpUnitsDiscLogSolver::new(3571);
    let base = 650;
    let x = solver
        .shanks_algorithm(base, 2213)
        .expect("Should have a logarithm");
    println!("log of {} base {} modulo 3571 is {}", 2213, base, x);
    println!("{}^{} = {}", base, x, solver.fast_power(base, x));
}
