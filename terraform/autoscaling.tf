# ECS 오토스케일링 타겟 설정
resource "aws_appautoscaling_target" "ecs_target" {
  max_capacity       = 3   # 최대 태스크 수
  min_capacity       = 1   # 최소 태스크 수
  resource_id        = "service/ops-copilot-cluster/ops-copilot-service"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"  # ECS 서비스
}

# CPU 기반 오토스케일링 정책
resource "aws_appautoscaling_policy" "ecs_cpu_policy" {
  name               = "ops-copilot-cpu-scaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.ecs_target.resource_id
  scalable_dimension = aws_appautoscaling_target.ecs_target.scalable_dimension
  service_namespace  = aws_appautoscaling_target.ecs_target.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70   # CPU 70% 목표
  }
}