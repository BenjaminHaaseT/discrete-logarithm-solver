use actix_web::body::BoxBody;
use actix_web::{
    error, get,
    http::{header::ContentType, StatusCode},
    post, web, App, HttpResponse, HttpServer, Responder,
};
use discrete_logarithm_lib::{is_prime, FpUnitsDiscLogSolver};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
struct FpInputs {
    prime: u32,
    base: u32,
    num: u32,
}

// #[derive(Debug, Deserialize)]
// struct SolverInputs {
//     base: u32,
//     num: u32,
// }

#[derive(Serialize)]
struct ShanksOutput {
    prime: u32,
    base: u32,
    num: u32,
    n: u32,
    log: u32,
    base_inverse: u32,
    collision_list1: HashMap<u32, i32>,
    collision_list2: HashMap<u32, i32>,
}

impl Responder for ShanksOutput {
    type Body = BoxBody;

    fn respond_to(self, req: &actix_web::HttpRequest) -> HttpResponse<Self::Body> {
        let body = serde_json::to_string(&self).unwrap();
        HttpResponse::Ok()
            .content_type(ContentType::json())
            .body(body)
    }
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

// struct AppSolver {
//     solver: Mutex<Option<FpUnitsDiscLogSolver>>,
// }

// impl AppSolver {
//     fn new() -> Self {
//         AppSolver {
//             solver: Mutex::new(None),
//         }
//     }
// }

// #[post("/new/{prime}")]
// async fn new_solver(
//     path: web::Path<u32>,
//     app_data: web::Data<AppSolver>,
// ) -> Result<HttpResponse, UserError> {
//     let mut guard = if let Ok(guard_option) = app_data.as_ref().solver.lock() {
//         guard_option
//     } else {
//         return Err(UserError::InternalServer);
//     };
//     // validate input from path
//     let prime = path.into_inner();
//     if !is_prime(prime) {
//         return Err(UserError::BadInput(format!("{} is not prime", prime)));
//     }
//     *guard = Some(FpUnitsDiscLogSolver::new(prime));
//     Ok(HttpResponse::Ok().body("created solver successfully\n"))
// }

// #[get("/modulus")]
// async fn get_modulus(app_data: web::Data<AppSolver>) -> Result<HttpResponse, UserError> {
//     let guard = if let Ok(guard_option) = app_data.as_ref().solver.lock() {
//         guard_option
//     } else {
//         return Err(UserError::InternalServer);
//     };

//     if let Some(ref fp) = *guard {
//         return Ok(HttpResponse::Ok().body(format!("prime modulus is {}\n", fp.prime)));
//     }

//     Err(UserError::BadInput(String::from(
//         "no solver has been created yet",
//     )))
// }

// #[get("/solve")]
// async fn solve_discrete_logarithm_with_solver(
//     data: web::Query<SolverInputs>,
//     app_data: web::Data<AppSolver>,
// ) -> Result<HttpResponse, UserError> {
//     // First get a lock on the mutex holding the solver
//     let guard = if let Ok(guard_option) = app_data.as_ref().solver.lock() {
//         guard_option
//     } else {
//         return Err(UserError::InternalServer);
//     };

//     // get a refernce to the solver if we have one
//     let solver = match guard.deref() {
//         Some(ref fp) => fp,
//         None => return Err(UserError::BadInput(String::from("no solver created yet"))),
//     };

//     // Get data from payload and validate it
//     let inputs = data.into_inner();
//     let (base, num) = (inputs.base, inputs.num);
//     if base % solver.prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "{} is not a valid base since {} mod {} = 0",
//             base, base, solver.prime
//         )));
//     } else if num % solver.prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "no logarithm for {} since {} mod {} = 0",
//             num, num, solver.prime
//         )));
//     }

//     if let Some(x) = solver.shanks_algorithm(base, num) {
//         return Ok(HttpResponse::Ok().body(x.to_string()));
//     }

//     Err(UserError::BadInput(format!(
//         "{} has not discrete logarithm with base {} modulo {}",
//         num, base, solver.prime
//     )))
// }

// #[get("/solve-with-output")]
// async fn solve_discrete_logarithm_with_solver_output(
//     data: web::Query<SolverInputs>,
//     app_data: web::Data<AppSolver>,
// ) -> Result<ShanksOutput, UserError> {
//     // Get lock on the mutex holding the solver
//     let guard = if let Ok(guard_opt) = app_data.as_ref().solver.lock() {
//         guard_opt
//     } else {
//         return Err(UserError::InternalServer);
//     };

//     // Validate we have a solver insantiated
//     let solver = match guard.deref() {
//         Some(fp) => fp,
//         None => return Err(UserError::BadInput(String::from("no solver created"))),
//     };

//     // Validate data sent from client
//     let (base, num) = (data.base, data.num);
//     if base % solver.prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "{} is not a valid base since {} mod {} = 0",
//             base, base, solver.prime
//         )));
//     } else if num % solver.prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "no logarithm for {} with base {} since {} mod {} = 0",
//             num, base, num, solver.prime
//         )));
//     }

//     let (log, base_inverse, n, list1, list2) = match solver.shanks_algorithm_with_output(base, num)
//     {
//         (Some(x), inv, ord, l1, l2) => (x, inv, ord, l1, l2),
//         (None, inv, ord, l1, l2) => (0, inv, ord, l1, l2),
//     };

//     Ok(ShanksOutput {
//         prime: solver.prime,
//         base,
//         num,
//         n,
//         log,
//         base_inverse,
//         collision_list1: list1,
//         collision_list2: list2,
//     })
// }

// #[get("/solve")]
// async fn solve_discrete_logarithm_with_output(
//     data: web::Json<Inputs>,
// ) -> Result<ShanksOutput, UserError> {
//     // Get input from json
//     let (prime, base, num) = (data.prime, data.base, data.num);
//     // Check input is valid
//     if !is_prime(prime) {
//         return Err(UserError::BadInput(format!("{} is not prime", prime)));
//     } else if base % prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "{} is not a valid base, {} mod {} = 0",
//             base, base, prime
//         )));
//     } else if num % prime == 0 {
//         return Err(UserError::BadInput(format!(
//             "{} has no logarithm mod {}, since {} mod {} = 0",
//             num, prime, num, prime
//         )));
//     }

//     // Get return data from solving discrete logarithm, if the log = 0 after matching we know no discrete logarithm exists
//     let (log, base_inverse, n, list1, list2) = match shanks_algorithm_with_output(prime, base, num)
//     {
//         (Some(l), inv, ord, l1, l2) => (l, inv, ord, l1, l2),
//         (None, inv, ord, l1, l2) => (0, inv, ord, l1, l2),
//     };

//     Ok(ShanksOutput {
//         prime,
//         base,
//         num,
//         log,
//         base_inverse,
//         n,
//         collision_list1: list1,
//         collision_list2: list2,
//     })
// }

/// A request handler that will solve the discrete logarithm given the supplied inputs via `info`. The handler will validate the inputs, if any validation fails `UserError` is returned.
/// TODO:!!
// #[get("solve-with-output")]
// async fn shanks_algorithm_with_output(
//     info: web::Query<FpInputs>,
// ) -> Result<ShanksOutput, UserError> {
// }

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
