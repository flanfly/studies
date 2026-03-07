locals {
  project_id = "prj-vertexai-test"
  region     = "asia-southeast1" # Singapore

  github_repo = "flanfly/studies"
  github_user = "flanfly"

  required_services = [
    "iam.googleapis.com",
    "iamcredentials.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "sts.googleapis.com",
    "batch.googleapis.com",
    "secretmanager.googleapis.com",
  ]

  roles = [
    "roles/batch.agentReporter",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/artifactregistry.reader",
    "roles/storage.objectAdmin",
    "roles/storage.admin",
    "roles/aiplatform.user",
  ]
}

terraform {
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
  project = local.project_id
  region  = local.region
}

data "google_project" "default" {
  project_id = local.project_id
}

resource "google_project_service" "enabled_apis" {
  for_each = toset(local.required_services)

  project = local.project_id
  service = each.key

  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "pipelines" {
  location      = local.region
  repository_id = "pipelines"
  format        = "KFP"
}

resource "google_artifact_registry_repository" "default" {
  location               = local.region
  repository_id          = "default"
  format                 = "DOCKER"
  cleanup_policy_dry_run = false

  vulnerability_scanning_config {
    enablement_config = "DISABLED"
  }
}

resource "google_service_account" "batch" {
  account_id   = "batch-job-sa"
  display_name = "Service Account for Batch Jobs"
}

resource "google_project_iam_member" "default" {
  for_each = toset(local.roles)

  project = local.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_secret_manager_secret" "git_key" {
  secret_id = "git-key"

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "git_key" {
  secret = google_secret_manager_secret.git_key.id

  secret_data = filebase64("${path.module}/../git.key")
}

resource "google_secret_manager_secret_iam_member" "default" {
  secret_id = google_secret_manager_secret.git_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.batch.email}"
}

resource "google_service_account" "github" {
  account_id   = "github-actions-deployer"
  display_name = "GitHub Actions Service Account"
}

resource "google_artifact_registry_repository_iam_member" "default" {
  location   = google_artifact_registry_repository.default.location
  repository = google_artifact_registry_repository.default.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.github.email}"
}

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github"
  display_name              = "GitHub Pool"
  description               = "Identity pool for GitHub Actions"
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github"
  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
  }
  attribute_condition = "assertion.repository_owner == 'flanfly'"
  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

resource "google_service_account_iam_member" "wif_impersonation" {
  service_account_id = google_service_account.github.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${local.github_repo}"
}

output "workload_identity_provider" {
  value       = google_iam_workload_identity_pool_provider.github.name
  description = "Use this value for 'workload_identity_provider' in your GitHub Action"
}

output "github_service_account_email" {
  value = google_service_account.github.email
}

output "batch_service_account_email" {
  value = google_service_account.batch.email
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.default.registry_uri
}

output "git_key" {
  value = google_secret_manager_secret_version.git_key.name
}
