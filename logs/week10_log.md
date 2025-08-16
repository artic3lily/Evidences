# Week 10 (Days 81–90) — CI/CD Automation & Cloud Deployment Config

**Focus:** Deployments, continuous integration pipelines, and release configuration management.  
**Deliverables:** GitHub Actions CI/CD pipeline configuration; production deployment parameters JSON config.

## Step-by-step

1) **CI/CD Pipeline Setup**
   - File: `.github/workflows/ci_cd.yml`
   - Configured GitHub Actions to automatically run on pushes to `main` and `master` branches.
   - Stage 1: Checkout code, set up Python environment, install requirements, lint python code with `ruff`, and compile all script modules to ensure syntactical correctness.
   - Stage 2: Trigger docker multi-platform build testing to verify Docker packaging.

2) **Environment Configurations**
   - File: `src/week10/deploy_config.json`
   - Established structured release configs detailing port mapping, replica counts, model settings, and local database storage directories.

3) **Final Repository Verification**
   - Completed end-to-end local testing.
   - Consolidated dependencies, verified renaming configurations, and completed log writing.

## Challenges and Resolutions
- **CI Dependency Size:** Storing local heavy embedding models inside docker images in CI can consume large space and block runner run times. Resolved by using standard Hugging Face pipelines that fetch model weights from cache or download on startup, keeping the Docker image small and lightweight.
