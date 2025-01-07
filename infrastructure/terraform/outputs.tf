# Network configuration outputs
output "network_config" {
  description = "Network configuration and security groups for GameGen-X infrastructure"
  value = {
    vpc_id           = module.network.vpc_id
    public_subnet    = module.network.public_subnet_id
    private_subnet   = module.network.private_subnet_id
    gpu_cluster_sg   = module.network.gpu_cluster_sg_id
    load_balancer_sg = module.network.lb_sg_id
    bastion_sg       = module.network.bastion_sg_id
    network_acls     = module.network.network_acls
    route_tables     = module.network.route_tables
  }
  sensitive = false
}

# Compute cluster configurations
output "compute_config" {
  description = "Compute cluster configurations for both FreeBSD and development environments"
  value = {
    freebsd_cluster = {
      cluster_id        = module.compute.freebsd_cluster_id
      instance_ids      = module.compute.freebsd_instance_ids
      gpu_node_count    = module.compute.gpu_node_count
      autoscaling_group = module.compute.asg_id
    }
    dev_cluster = {
      cluster_id          = module.compute.dev_cluster_id
      gpu_nodes          = module.compute.gpu_node_ids
      training_nodes     = module.compute.training_node_ids
      monitoring_endpoint = module.compute.monitoring_endpoint
    }
  }
  sensitive = true
}

# Storage configurations
output "storage_config" {
  description = "Storage configurations for video data, model weights, and training data"
  value = {
    object_storage = {
      bucket_name    = module.storage.video_bucket_name
      endpoint       = module.storage.minio_endpoint
      access_policy  = module.storage.bucket_policy
    }
    model_storage = {
      fs_id         = module.storage.model_fs_id
      mount_targets = module.storage.fs_mount_targets
      backup_vault  = module.storage.backup_vault_id
    }
    training_data = {
      volume_id          = module.storage.training_volume_id
      snapshot_config    = module.storage.snapshot_config
      replication_status = module.storage.replication_status
    }
  }
  sensitive = true
}