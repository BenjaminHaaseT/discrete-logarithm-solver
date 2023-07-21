use actix_web::body::BoxBody;
use actix_web::{
    error, get,
    http::{header::ContentType, StatusCode},
    post, web, App, HttpResponse, HttpServer, Responder,
};
use discrete_logarithm_lib::{is_prime, FpUnitsDiscLogSolver, ShanksOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct FpInputs {
    prime: u32,
    base: u32,
    num: u32,
}

#[derive(Debug)]
enum UserError {
    BadInput(String),
    InternalServer,
    NoSolution(String),
}

impl std::fmt::Display for UserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            UserError::BadInput(ref s) => write!(f, "{s}"),
            UserError::NoSolution(ref s) => write!(f, "{s}"),
            UserError::InternalServer => {
                write!(f, "An internal error occured. Please try again later.")
            }
        }
    }
}

impl error::ResponseError for UserError {
    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code()).body(self.to_string())
    }

    fn status_code(&self) -> StatusCode {
        match *self {
            UserError::BadInput(_) => StatusCode::BAD_REQUEST,
            UserError::NoSolution(_) => StatusCode::CONFLICT,
            UserError::InternalServer => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

/// A handler that will take a query from the caller, validate that the inputs sent meets the constraints, then solve the discrete logarithm.
/// Returns an `HttpResponse::Ok()` if the discrete logarithm is found, otherwise it returns a `UserError::NoSolution`.
/// It is up to the caller to determine what to do in the case where no solution exists.
///
#[get("/shanks-algorithm")]
async fn shanks_algorithm_handler(info: web::Query<FpInputs>) -> Result<HttpResponse, UserError> {
    let (prime, base, num) = (info.prime, info.base, info.num);
    // Validate prime number, base and num
    if !is_prime(prime) {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not prime",
            prime
        )));
    } else if base % prime == 0 {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not a valid base since {} mod {} = 0",
            base, base, prime
        )));
    } else if num % prime == 0 {
        return Err(UserError::BadInput(format!(
            "{} has no logarithm with base {} since {} mod {} = 0",
            num, base, num, prime,
        )));
    }

    let solver = FpUnitsDiscLogSolver::new(prime);
    if let Some(res) = solver.shanks_algorithm(base, num) {
        return Ok(HttpResponse::Ok().body(res.to_string()));
    }

    Err(UserError::BadInput(format!(
        "{} has no logarithm base {}",
        num, base
    )))
}

/// Handler for solving the discrete logarithm using shanks algorithm that also returns the interesting details of the computation,
///  e.g. the lists created when searching for a collision etc... as output via a `ShanksOutput`.
#[get("/shanks-algorithm-with-output")]
async fn shanks_algorithm_with_output_handler(
    info: web::Query<FpInputs>,
) -> Result<ShanksOutput, UserError> {
    let (prime, base, num) = (info.prime, info.base, info.num);
    // Validate the inputs
    if !is_prime(prime) {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not prime",
            prime
        )));
    } else if base % prime == 0 {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not a valid base since {} mod {} = 0",
            base, base, prime
        )));
    } else if num % prime == 0 {
        return Err(UserError::BadInput(format!(
            "{} has no logarithm with base {} since {} mod {} = 0",
            num, base, num, prime,
        )));
    }

    let solver = FpUnitsDiscLogSolver::new(prime);

    // perform computation
    match solver.shanks_algorithm_with_output(base, num) {
        (Some(log), u, n, collision_list1, collision_list2) => {
            return Ok(ShanksOutput {
                prime: solver.prime,
                base,
                num,
                n,
                log,
                base_inverse: u,
                collision_list1,
                collision_list2,
            });
        }
        (None, _u, _n, _collision_list1, _collision_list2) => {
            return Err(UserError::NoSolution(format!(
                "no solution for ({}, {})",
                base, num
            )));
        }
    }
}

#[get("/pollhig-hellman")]
async fn pollhig_hellman_handler(input: web::Query<FpInputs>) -> Result<HttpResponse, UserError> {
    // Validate inputs first
    let (prime, base, num) = (input.prime, input.base, input.num);

    if !is_prime(prime) {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not prime",
            prime
        )));
    } else if base % prime == 0 {
        return Err(UserError::BadInput(format!(
            "invalid input, {} is not a valid base since {} mod {} = 0",
            base, base, prime
        )));
    } else if num % prime == 0 {
        return Err(UserError::BadInput(format!(
            "no solution since {} mod {} = 0",
            num, prime
        )));
    }

    let solver = FpUnitsDiscLogSolver::new(prime);

    if let Some(solution) = solver.pollhig_hellman(base, num) {
        return Ok(HttpResponse::Ok().body(solution.to_string()));
    }

    Err(UserError::NoSolution(String::from(
        "no solution for given inputs",
    )))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let address = "127.0.0.1";
    let port = 8080;

    HttpServer::new(|| {
        App::new().service(
            web::scope("/shanks")
                .service(shanks_algorithm_handler)
                .service(shanks_algorithm_with_output_handler),
        )
    })
    .bind((address, port))?
    .run()
    .await
}
