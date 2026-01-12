locals {
  project_id = "prj-vertexai-test"
  region     = "asia-southeast1" # Singapore
  github_repo = "flanfly/studies"
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

import {
  id = "projects/prj-vertexai-test/locations/asia-southeast1/repositories/default"
  to = google_artifact_registry_repository.default
}

resource "google_artifact_registry_repository" "default" {
  location      = local.region
  repository_id = "default"
  format        = "DOCKER"
  cleanup_policy_dry_run = true

  docker_config {
    immutable_tags = false
  }
}

resource "google_service_account" "default" {
  account_id   = "github-actions-deployer"
  display_name = "GitHub Actions Service Account"
}

resource "google_artifact_registry_repository_iam_member" "default" {
  location   = google_artifact_registry_repository.default.location
  repository = google_artifact_registry_repository.default.name
  role       = "roles/artifactregistry.writer"
  member     = "serviceAccount:${google_service_account.default.email}"
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
  service_account_id = google_service_account.default.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.repository/${local.github_repo}"
}

output "workload_identity_provider" {
  value = google_iam_workload_identity_pool_provider.github.name
  description = "Use this value for 'workload_identity_provider' in your GitHub Action"
}

output "service_account_email" {
  value = google_service_account.default.email
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.default.registry_uri
}
