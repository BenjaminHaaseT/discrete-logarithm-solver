use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use discrete_logarithm_lib::{is_prime, FpUnitsDiscLogSolver};
use std::sync::{Arc, Mutex};

struct AppSolver {
    solver: Mutex<Option<FpUnitsDiscLogSolver>>,
}

impl AppSolver {
    fn new() -> Self {
        AppSolver {
            solver: Mutex::new(None),
        }
    }
}

#[post("/new/{prime}")]
async fn new_solver(path: web::Path<u32>, data: web::Data<AppSolver>) -> impl Responder {
    let mut guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
        guard_option
    } else {
        return HttpResponse::Conflict().finish();
    };
    *guard = Some(FpUnitsDiscLogSolver::new(path.into_inner()));
    HttpResponse::Ok().body("created solver successfully\n")
}

#[get("/modulus")]
async fn get_modulus(data: web::Data<AppSolver>) -> impl Responder {
    let guard = if let Ok(guard_option) = data.as_ref().solver.lock() {
        guard_option
    } else {
        return HttpResponse::Conflict().finish();
    };

    if let Some(ref fp) = *guard {
        return HttpResponse::Ok().body(format!("prime modulus is {}\n", fp.prime));
    }

    HttpResponse::Conflict().body("no solver created yet\n")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let address = "127.0.0.1";
    let port = 8080;
    println!("connecting to {}:{}...", address, port);

    let app_solver = web::Data::new(AppSolver::new());

    HttpServer::new(move || {
        App::new()
            .app_data(app_solver.clone())
            .service(new_solver)
            .service(get_modulus)
    })
    .bind((address, port))?
    .run()
    .await
}
