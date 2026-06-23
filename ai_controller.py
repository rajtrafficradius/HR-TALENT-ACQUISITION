"""Minimal AI behaviors for the first playable slice."""

from __future__ import annotations

import math

from src.combat_logic import launch_salvo
from src.enums import ContactQuality, EngagementDoctrine, WeaponCategory
from src.utils import add_vec, chance, distance_km, heading_to_vector, normalize, scale_vec, sub_vec


class AIController:
    """Surface combat AI with conservative salvo logic."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, object]] = {}

    def update(self, engine, dt_seconds: float) -> None:
        del dt_seconds
        self._coordinate_forces(engine)
        self._offensive_fire(engine)
        self._defensive_fire(engine)

    def _platform_state(self, platform_id: str) -> dict[str, object]:
        return self._state.setdefault(
            platform_id,
            {
                "next_salvo_time_s": 0.0,
                "next_sam_time_s": 0.0,
                "support_until_s": 0.0,
                "evade_until_s": 0.0,
                "next_nav_update_s": 0.0,
                "last_waypoint": None,
            },
        )

    def _offensive_fire(self, engine) -> None:
        for platform in engine.platforms.values():
            if not platform.alive:
                continue
            state = self._platform_state(platform.entity_id)
            if engine.current_time_s < state["next_salvo_time_s"]:
                continue
            if platform.platform_type == "fighter" and self._fighter_multi_engage(engine, platform, state):
                continue
            air_solution = self._pick_air_engagement(engine, platform)
            visual_air_target = None
            if air_solution is not None:
                air_contact, air_target = air_solution
            else:
                air_contact = None
                air_target = self._pick_visual_air_target(engine, platform) if platform.platform_type == "fighter" else None
                visual_air_target = air_target
            if air_target is not None:
                air_weapon = self._pick_air_weapon(platform, engine, air_target)
                if air_weapon is not None:
                    weapon_template = engine.inventory.get_weapon(air_weapon)
                    current_total = self._remaining_category_ammo(platform, engine, WeaponCategory.AIR_TO_AIR)
                    reserve = self._fighter_air_reserve(engine, platform) if platform.platform_type == "fighter" else 0
                    reserve_locked = (
                        platform.platform_type == "fighter"
                        and air_target.platform_type == "fighter"
                        and not self._fighter_should_spend_reserve(engine, platform, air_target)
                    )
                    if (
                        reserve_locked
                        and current_total <= reserve
                    ):
                        continue
                    if self._existing_air_commitments(engine, platform.side.value, air_target.entity_id) >= self._desired_air_commitments(air_target):
                        continue
                    attack_factor = 1.0 if platform.platform_type == "fighter" else 0.98 if platform.scenario_role != "fighter_strike" else 0.55
                    if air_contact is not None and air_contact.quality == ContactQuality.SEARCH:
                        attack_factor *= 0.88
                    if air_contact is not None and air_contact.source_sensor.startswith("network:"):
                        attack_factor = max(attack_factor, 0.95)
                    attack_factor *= self._engagement_factor(air_target)
                    if distance_km(platform.position, air_target.position) <= weapon_template.max_range_km * attack_factor and self._within_attack_cone(
                        platform,
                        air_target,
                        allow_offboard=bool(air_contact is not None and air_contact.source_sensor.startswith("network:")),
                        quality=air_contact.quality if air_contact is not None else ContactQuality.FIRE_CONTROL,
                    ):
                        guidance_label = self._guidance_label(engine, platform, air_contact, air_target) if air_contact is not None else ""
                        salvo_count = self._fighter_air_salvo_count(engine, platform, air_weapon, air_target) if platform.platform_type == "fighter" else 1
                        if reserve_locked:
                            salvo_count = min(salvo_count, max(0, current_total - reserve))
                        if salvo_count <= 0:
                            continue
                        launched = launch_salvo(
                            engine,
                            platform,
                            air_target.entity_id,
                            air_weapon,
                            count=salvo_count,
                            metadata_updates={"guidance_label": guidance_label} if guidance_label else None,
                        )
                        if launched:
                            state["guided_target_id"] = air_target.entity_id
                            state["support_until_s"] = engine.current_time_s + 18.0
                            state["next_salvo_time_s"] = engine.current_time_s + 6.0
                            continue
                if platform.platform_type == "fighter" and self._gun_attack(engine, platform, air_target):
                    state["next_salvo_time_s"] = engine.current_time_s + 2.0
                    continue
            target = self._pick_target(engine, platform)
            if target is None:
                continue
            offensive_weapon = self._pick_offensive_weapon(engine, platform, target)
            if offensive_weapon is None:
                continue
            weapon_template = engine.inventory.get_weapon(offensive_weapon)
            contact = platform.contacts.get(target.entity_id)
            network_cue = bool(contact is not None and contact.source_sensor.startswith("network:"))
            if distance_km(platform.position, target.position) > weapon_template.max_range_km * self._engagement_factor(target) * (1.0 if network_cue else 0.95):
                continue
            reserve = self._offensive_reserve(engine, platform, offensive_weapon)
            remaining = platform.weapon_stock.get(offensive_weapon, 0) - reserve
            if remaining <= 0:
                continue
            if self._existing_surface_commitments(engine, platform.side.value, target.entity_id) >= self._desired_surface_commitments(target):
                continue
            salvo_count = 2 if platform.doctrine == EngagementDoctrine.DEFEND_CARRIER else 4
            if platform.platform_type == "fighter":
                salvo_count = 1
            elif platform.platform_type == "submarine":
                salvo_count = 2 if weapon_template.category == WeaponCategory.TORPEDO else 1
            salvo_count = max(1, min(salvo_count, remaining))
            surface_guidance = self._surface_guidance_label(engine, platform, target)
            launched = launch_salvo(
                engine,
                platform,
                target.entity_id,
                offensive_weapon,
                count=salvo_count,
                metadata_updates={"guidance_label": surface_guidance} if surface_guidance else None,
            )
            if launched:
                if platform.platform_type == "fighter":
                    cooldown = 8.0 if platform.scenario_role == "fighter_strike" else 10.0
                elif platform.platform_type == "submarine":
                    cooldown = 55.0 if weapon_template.category == WeaponCategory.ANTI_SHIP else 40.0
                else:
                    cooldown = 24.0 if platform.scenario_role == "fighter_strike" else 28.0 if platform.doctrine == EngagementDoctrine.SEARCH_AND_DESTROY else 40.0
                state["next_salvo_time_s"] = engine.current_time_s + cooldown

    def _defensive_fire(self, engine) -> None:
        platforms_list = list(engine.platforms.values())
        ordered_platforms = sorted(
            (platform for platform in engine.platforms.values() if platform.alive),
            key=lambda item: (item.side.value, self._nearest_defense_threat_distance(engine, item), self._defense_priority_rank(item)),
        )
        for platform in ordered_platforms:
            if not platform.alive:
                continue
            state = self._platform_state(platform.entity_id)
            if engine.current_time_s < state["next_sam_time_s"]:
                continue
            clutter = engine.config.clutter_factor * engine.scenario_context.get("weather_factor", 1.0)
            projectile_launches = 0
            air_launches = 0
            projectile_budget = self._projectile_defense_budget(platform)
            projectile_threats = []
            air_defense_targets = self._ship_air_defense_targets(engine, platform)
            for projectile in engine.projectiles.values():
                if not projectile.alive or projectile.side == platform.side or projectile.defensive:
                    continue
                missile_flight_profile = projectile.metadata.get("flight_profile", "cruise")
                projectile_distance = distance_km(platform.position, projectile.position)
                local_weapon = self._pick_projectile_defense_weapon(platform, engine, projectile, projectile_distance)
                if local_weapon is None:
                    continue
                local_template = engine.inventory.get_weapon(local_weapon)
                same_side_target = projectile.target_id in engine.platforms and engine.platforms[projectile.target_id].side == platform.side
                fleet_ballistic_defense = missile_flight_profile == "ballistic" and same_side_target
                fleet_cruise_defense = missile_flight_profile != "ballistic" and same_side_target
                fighter_point_defense = (
                    platform.platform_type == "fighter"
                    and projectile.category == WeaponCategory.ANTI_SHIP
                    and projectile_distance <= local_template.max_range_km * 0.25
                )
                if projectile.target_id != platform.entity_id and not fleet_ballistic_defense and not fleet_cruise_defense and not fighter_point_defense:
                    continue
                local_detection_window = engine.detector.projectile_detection_range_km(platform, projectile, clutter)
                network_detection_window, network_source = engine.detector.network_projectile_cue(platform, platforms_list, projectile, clutter)
                if platform.platform_type == "fighter":
                    engage_window = min(local_template.max_range_km * 0.25, max(8.0, local_detection_window, network_detection_window))
                elif missile_flight_profile == "ballistic":
                    engage_window = min(local_template.max_range_km * 0.72, max(local_detection_window, network_detection_window) + 60.0)
                else:
                    engage_window = min(local_template.max_range_km * 0.55, max(local_detection_window, network_detection_window) + 8.0)
                if projectile_distance > engage_window:
                    continue
                target = engine.platforms.get(projectile.target_id)
                target_priority = 0
                if target is not None and target.platform_type == "carrier":
                    target_priority = 2
                elif target is not None and target.platform_type in {"cruiser", "destroyer"}:
                    target_priority = 1
                projectile_threats.append(
                    (
                        projectile.target_id != platform.entity_id,
                        missile_flight_profile != "ballistic",
                        -target_priority,
                        projectile_distance,
                        projectile,
                        local_weapon,
                        local_template,
                        network_source,
                    )
                )
            projectile_threats.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
            for _offboard, _non_ballistic, _target_priority, _distance, projectile, local_weapon, local_template, network_source in projectile_threats:
                if projectile_launches >= projectile_budget or local_weapon is None:
                    break
                if platform.weapon_stock.get(local_weapon, 0) <= self._weapon_reserve(engine, platform, local_weapon):
                    continue
                current_commitments = self._existing_defensive_commitments(engine, platform.side.value, projectile.entity_id)
                desired_commitments = self._desired_projectile_commitments(engine, projectile)
                pending = min(projectile_budget - projectile_launches, desired_commitments - current_commitments)
                if pending <= 0:
                    continue
                guidance = network_source if network_source and distance_km(platform.position, projectile.position) > engine.detector.projectile_detection_range_km(platform, projectile, clutter) else ""
                launched = launch_salvo(
                    engine,
                    platform,
                    projectile.entity_id,
                    local_weapon,
                    count=pending,
                    metadata_updates={"guidance_label": guidance} if guidance else None,
                )
                projectile_launches += launched
            air_budget = self._air_defense_budget(platform)
            if air_budget > 0:
                for air_contact, air_target in air_defense_targets:
                    if air_launches >= air_budget:
                        break
                    engagement_weapon = self._pick_ship_air_weapon(platform, engine, air_target, air_contact.distance_km)
                    if engagement_weapon is None:
                        continue
                    if platform.weapon_stock.get(engagement_weapon, 0) <= self._weapon_reserve(engine, platform, engagement_weapon):
                        continue
                    engagement_template = engine.inventory.get_weapon(engagement_weapon)
                    if air_contact.distance_km > engagement_template.max_range_km * self._engagement_factor(air_target):
                        continue
                    current_commitments = self._existing_defensive_commitments(engine, platform.side.value, air_target.entity_id)
                    desired_commitments = self._desired_air_defense_commitments(engine, air_target, air_contact)
                    pending = min(air_budget - air_launches, desired_commitments - current_commitments)
                    if pending <= 0:
                        continue
                    launched = launch_salvo(
                        engine,
                        platform,
                        air_target.entity_id,
                        engagement_weapon,
                        count=pending,
                        metadata_updates={"guidance_label": self._guidance_label(engine, platform, air_contact, air_target)} if air_contact.source_sensor.startswith("network:") else None,
                    )
                    air_launches += launched
            if projectile_launches or air_launches:
                persistent_threats = bool(projectile_threats or air_defense_targets)
                if platform.platform_type == "fighter":
                    state["next_sam_time_s"] = engine.current_time_s + (1.0 if persistent_threats else 2.0)
                else:
                    state["next_sam_time_s"] = engine.current_time_s + (1.0 if persistent_threats else 3.0)

    def _pick_target(self, engine, platform):
        if platform.doctrine == EngagementDoctrine.DEFEND_CARRIER and not platform.assigned_guard_target_id:
            same_side = [ship for ship in engine.platforms.values() if ship.side == platform.side and ship.entity_id != platform.entity_id and ship.alive]
            if same_side:
                same_side.sort(key=lambda ship: ship.max_hull_points, reverse=True)
                platform.assigned_guard_target_id = same_side[0].entity_id
        if platform.doctrine == EngagementDoctrine.DEFEND_CARRIER:
            threats = [
                projectile for projectile in engine.projectiles.values()
                if projectile.alive and projectile.side != platform.side and projectile.category == WeaponCategory.ANTI_SHIP
            ]
            if threats and platform.assigned_guard_target_id in engine.platforms:
                guard_target = engine.platforms[platform.assigned_guard_target_id]
                close_threats = [threat for threat in threats if threat.target_id == guard_target.entity_id]
                if close_threats:
                    return guard_target
        allowed_qualities = self._allowed_contact_qualities(platform)
        valid_contacts = [
            (contact.distance_km, engine.platforms[contact.target_id])
            for contact in platform.contacts.values()
            if contact.quality in allowed_qualities
            and contact.target_id in engine.platforms
            and engine.platforms[contact.target_id].alive
            and engine.platforms[contact.target_id].platform_type not in {"fighter", "awacs", "helicopter"}
        ]
        if not valid_contacts:
            return None
        if platform.doctrine == EngagementDoctrine.SEARCH_AND_DESTROY:
            valid_contacts.sort(key=lambda item: (engine.platforms[item[1].entity_id].damage_state.level.value == "mission_killed", item[0]))
        else:
            valid_contacts.sort(key=lambda item: item[0])
        return valid_contacts[0][1]

    def _allowed_contact_qualities(self, platform) -> set[ContactQuality]:
        if platform.doctrine == EngagementDoctrine.FIRST_CONTACT:
            return {ContactQuality.SEARCH, ContactQuality.TRACK, ContactQuality.FIRE_CONTROL}
        if platform.doctrine == EngagementDoctrine.SEARCH_AND_DESTROY:
            # Allow offboard AWACS/sensor-net cueing to trigger long-range shots.
            if any(contact.source_sensor.startswith("network:") for contact in platform.contacts.values()):
                return {ContactQuality.SEARCH, ContactQuality.TRACK, ContactQuality.FIRE_CONTROL}
        if platform.platform_type in {"fighter", "submarine"}:
            return {ContactQuality.SEARCH, ContactQuality.TRACK, ContactQuality.FIRE_CONTROL}
        return {ContactQuality.TRACK, ContactQuality.FIRE_CONTROL}

    def _pick_weapon(self, platform, engine, category: WeaponCategory) -> str | None:
        for weapon_id, count in platform.weapon_stock.items():
            if count <= 0:
                continue
            if engine.inventory.get_weapon(weapon_id).category == category:
                return weapon_id
        return None

    def _is_ballistic_weapon(self, engine, weapon_id: str) -> bool:
        return engine.inventory.get_weapon(weapon_id).flight_profile == "ballistic"

    def _pick_interceptor_weapon(self, platform, engine, anti_ballistic_only: bool = False) -> str | None:
        candidates: list[tuple[float, str]] = []
        for weapon_id, count in platform.weapon_stock.items():
            if count <= self._weapon_reserve(engine, platform, weapon_id) or weapon_id not in engine.inventory.weapons:
                continue
            weapon = engine.inventory.get_weapon(weapon_id)
            if weapon.category != WeaponCategory.SAM:
                continue
            if anti_ballistic_only and not weapon.anti_ballistic:
                continue
            if not anti_ballistic_only and weapon.anti_ballistic:
                continue
            candidates.append((weapon.max_range_km, weapon_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _weapon_reserve(self, engine, platform, weapon_id: str) -> int:
        if weapon_id not in engine.inventory.weapons:
            return 0
        weapon = engine.inventory.get_weapon(weapon_id)
        if weapon.category != WeaponCategory.SAM:
            return 0
        template_total = engine.inventory.get_platform(platform.template_id).weapon_loadout.get(weapon_id, 0)
        if template_total <= 0:
            return 0
        return int(template_total * platform.sam_reserve_pct)

    def _pick_available_sam_weapon(self, platform, engine, min_range_km: float = 0.0, anti_ballistic_only: bool = False) -> str | None:
        candidates: list[tuple[float, int, str]] = []
        for weapon_id, count in platform.weapon_stock.items():
            if weapon_id not in engine.inventory.weapons:
                continue
            reserve = self._weapon_reserve(engine, platform, weapon_id)
            if count <= reserve:
                continue
            weapon = engine.inventory.get_weapon(weapon_id)
            if weapon.category != WeaponCategory.SAM:
                continue
            if anti_ballistic_only and not weapon.anti_ballistic:
                continue
            if not anti_ballistic_only and weapon.anti_ballistic:
                continue
            if weapon.max_range_km < min_range_km:
                continue
            candidates.append((weapon.max_range_km, count - reserve, weapon_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return candidates[0][2]

    def _pick_projectile_defense_weapon(self, platform, engine, projectile, projectile_distance_km: float) -> str | None:
        if platform.platform_type == "fighter":
            return self._pick_air_weapon(platform, engine, projectile)
        clutter_factor = engine.config.clutter_factor * engine.scenario_context.get("weather_factor", 1.0)
        candidates: list[tuple[float, str]] = []
        anti_ballistic = projectile.metadata.get("flight_profile") == "ballistic"
        target = engine.platforms.get(projectile.target_id)
        high_value = target is not None and target.platform_type in {"carrier", "cruiser", "destroyer", "launcher"}
        for weapon_id, count in platform.weapon_stock.items():
            if weapon_id not in engine.inventory.weapons:
                continue
            reserve = self._weapon_reserve(engine, platform, weapon_id)
            if count <= reserve:
                continue
            weapon = engine.inventory.get_weapon(weapon_id)
            if weapon.category != WeaponCategory.SAM:
                continue
            if anti_ballistic and not weapon.anti_ballistic:
                continue
            if not anti_ballistic and weapon.anti_ballistic:
                continue
            local_detection_window = engine.detector.projectile_detection_range_km(platform, projectile, clutter_factor)
            network_detection_window, _ = engine.detector.network_projectile_cue(platform, list(engine.platforms.values()), projectile, clutter_factor)
            if platform.platform_type == "fighter":
                engage_window = min(weapon.max_range_km * 0.25, max(8.0, local_detection_window, network_detection_window))
            elif anti_ballistic:
                engage_window = min(weapon.max_range_km * 0.72, max(local_detection_window, network_detection_window) + 60.0)
            else:
                engage_window = min(weapon.max_range_km * 0.55, max(local_detection_window, network_detection_window) + 8.0)
            if projectile_distance_km > engage_window:
                continue
            coverage = projectile_distance_km / max(engage_window, 1.0)
            score = weapon.hit_probability * 120.0 + min(24.0, (count - reserve) * 2.0)
            score += max(0.0, 90.0 - abs(0.62 - coverage) * 150.0)
            if anti_ballistic:
                score += 180.0 if weapon.anti_ballistic else -250.0
            elif high_value:
                score += 18.0
            if weapon.template_id == "sm6" and coverage >= 0.45:
                score += 24.0
            if coverage < 0.28 and weapon.max_range_km > 160.0 and not high_value:
                score -= 18.0
            score += min(20.0, 10.0 * float(projectile.metadata.get("failed_intercepts", 0)))
            candidates.append((score, weapon_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _pick_ship_air_weapon(self, platform, engine, target, target_distance_km: float | None = None) -> str | None:
        if platform.platform_type == "fighter":
            return self._pick_air_weapon(platform, engine, target)
        separation = target_distance_km if target_distance_km is not None else distance_km(platform.position, target.position)
        candidates: list[tuple[float, str]] = []
        for weapon_id, count in platform.weapon_stock.items():
            if weapon_id not in engine.inventory.weapons:
                continue
            reserve = self._weapon_reserve(engine, platform, weapon_id)
            if count <= reserve:
                continue
            weapon = engine.inventory.get_weapon(weapon_id)
            if weapon.category != WeaponCategory.SAM or weapon.anti_ballistic or weapon.max_range_km < separation:
                continue
            coverage = separation / max(weapon.max_range_km, 1.0)
            score = weapon.hit_probability * 100.0 + min(20.0, (count - reserve) * 2.0)
            score += max(0.0, 100.0 - abs(0.58 - coverage) * 140.0)
            if target.platform_type == "awacs":
                score += 22.0
            if getattr(target, "scenario_role", "") == "fighter_strike":
                score += 18.0
            if weapon.template_id == "sm6" and coverage >= 0.42:
                score += 22.0
            if coverage < 0.25 and weapon.max_range_km > 160.0 and target.platform_type != "awacs":
                score -= 14.0
            candidates.append((score, weapon_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _pick_air_weapon(self, platform, engine, target) -> str | None:
        candidates = [
            (engine.inventory.get_weapon(weapon_id).max_range_km, weapon_id)
            for weapon_id, count in platform.weapon_stock.items()
            if count > 0 and weapon_id in engine.inventory.weapons and engine.inventory.get_weapon(weapon_id).category == WeaponCategory.AIR_TO_AIR
        ]
        if not candidates:
            return None
        separation = distance_km(platform.position, target.position)
        if separation <= 55.0:
            close_candidates = [candidate for candidate in candidates if candidate[0] >= separation * 0.9]
            if close_candidates:
                close_candidates.sort(key=lambda item: item[0])
                return close_candidates[0][1]
        in_range = [candidate for candidate in candidates if candidate[0] >= separation * 0.78]
        if in_range:
            in_range.sort(key=lambda item: item[0], reverse=True)
            return in_range[0][1]
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _pick_offensive_weapon(self, engine, platform, target) -> str | None:
        if target.platform_type == "submarine":
            torpedo = self._pick_weapon(platform, engine, WeaponCategory.TORPEDO)
            if torpedo is not None:
                return torpedo
            return None
        if platform.platform_type == "submarine":
            torpedo = self._pick_weapon(platform, engine, WeaponCategory.TORPEDO)
            anti_ship = self._pick_weapon(platform, engine, WeaponCategory.ANTI_SHIP)
            if torpedo is not None:
                torpedo_range = engine.inventory.get_weapon(torpedo).max_range_km
                if distance_km(platform.position, target.position) <= torpedo_range * 0.88:
                    return torpedo
            return anti_ship or torpedo
        separation = distance_km(platform.position, target.position)
        contact = platform.contacts.get(target.entity_id)
        network_cue = bool(contact is not None and contact.source_sensor.startswith("network:"))
        candidates: list[tuple[float, str]] = []
        for weapon_id, count in platform.weapon_stock.items():
            if count <= 0 or weapon_id not in engine.inventory.weapons:
                continue
            weapon = engine.inventory.get_weapon(weapon_id)
            if weapon.category != WeaponCategory.ANTI_SHIP:
                continue
            if weapon.flight_profile == "ballistic" and not platform.ballistic_engagement_enabled:
                continue
            reserve = self._offensive_reserve(engine, platform, weapon_id)
            if count <= reserve:
                continue
            engage_limit = weapon.max_range_km * self._engagement_factor(target) * (1.0 if network_cue else 0.95)
            if separation > engage_limit:
                continue
            coverage = separation / max(weapon.max_range_km, 1.0)
            score = weapon.hit_probability * 90.0 + min(24.0, (count - reserve) * 2.0)
            score += max(0.0, 120.0 - abs(0.62 - coverage) * 160.0)
            if weapon.flight_profile == "ballistic":
                if target.platform_type in {"carrier", "cruiser", "destroyer", "launcher"}:
                    score += 220.0
                if separation < 180.0:
                    score -= 120.0
            if weapon.template_id == "tomahawk_mst":
                if network_cue:
                    score += 90.0
                if separation >= 320.0:
                    score += 120.0
                if separation < 220.0:
                    score -= 70.0
            if target.platform_type in {"carrier", "cruiser", "destroyer", "launcher"}:
                score += 32.0
            if coverage < 0.25 and weapon.max_range_km > 600.0 and not network_cue:
                score -= 36.0
            candidates.append((score, weapon_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _offensive_reserve(self, engine, platform, weapon_id: str) -> int:
        if platform.platform_type in {"fighter", "submarine", "awacs", "helicopter"}:
            return 0
        template_total = engine.inventory.get_platform(platform.template_id).weapon_loadout.get(weapon_id, 0)
        if template_total <= 2:
            return 0
        return max(1, int(round(template_total * 0.1)))

    def _remaining_category_ammo(self, platform, engine, category: WeaponCategory) -> int:
        return sum(
            count
            for weapon_id, count in platform.weapon_stock.items()
            if count > 0 and engine.inventory.get_weapon(weapon_id).category == category
        )

    def _fighter_multi_engage(self, engine, platform, state) -> bool:
        if platform.platform_type != "fighter":
            return False
        guidance_channels = 0
        launched_any = False
        targets_used: set[str] = set()
        current_total = self._remaining_category_ammo(platform, engine, WeaponCategory.AIR_TO_AIR)
        reserve = self._fighter_air_reserve(engine, platform)
        for air_contact, air_target in self._air_engagement_solutions(engine, platform):
            if guidance_channels >= self._fighter_engagement_budget(engine, platform, current_total):
                break
            if air_target.entity_id in targets_used:
                continue
            air_weapon = self._pick_air_weapon(platform, engine, air_target)
            if air_weapon is None:
                continue
            reserve_locked = (
                air_target.platform_type == "fighter"
                and not self._fighter_should_spend_reserve(engine, platform, air_target)
            )
            if (
                reserve_locked
                and current_total <= reserve
            ):
                continue
            if self._existing_air_commitments(engine, platform.side.value, air_target.entity_id) >= self._desired_air_commitments(air_target):
                continue
            weapon_template = engine.inventory.get_weapon(air_weapon)
            attack_factor = 1.0
            if air_contact.quality == ContactQuality.SEARCH:
                attack_factor *= 0.88
            if air_contact.source_sensor.startswith("network:"):
                attack_factor = max(attack_factor, 0.95)
            attack_factor *= self._engagement_factor(air_target)
            if distance_km(platform.position, air_target.position) > weapon_template.max_range_km * attack_factor:
                continue
            if not self._within_attack_cone(
                platform,
                air_target,
                allow_offboard=air_contact.source_sensor.startswith("network:"),
                quality=air_contact.quality,
            ):
                continue
            salvo_count = self._fighter_air_salvo_count(engine, platform, air_weapon, air_target)
            available_to_fire = current_total if not reserve_locked else max(0, current_total - reserve)
            salvo_count = min(salvo_count, available_to_fire)
            if salvo_count <= 0:
                continue
            guidance_label = self._guidance_label(engine, platform, air_contact, air_target)
            launched = launch_salvo(
                engine,
                platform,
                air_target.entity_id,
                air_weapon,
                count=salvo_count,
                metadata_updates={"guidance_label": guidance_label} if guidance_label else None,
            )
            if launched:
                current_total -= launched
                guidance_channels += 1
                launched_any = True
                targets_used.add(air_target.entity_id)
                state["guided_target_id"] = air_target.entity_id
                state["support_until_s"] = engine.current_time_s + 18.0
        if launched_any:
            state["next_salvo_time_s"] = engine.current_time_s + 3.0
            return True
        gun_target = self._pick_visual_air_target(engine, platform)
        if gun_target is not None and self._gun_attack(engine, platform, gun_target):
            platform.air_state = "dogfight"
            state["guided_target_id"] = ""
            state["support_until_s"] = 0.0
            state["next_salvo_time_s"] = engine.current_time_s + 2.0
            return True
        return False

    def _fighter_engagement_budget(self, engine, platform, current_total: int) -> int:
        if current_total <= 0:
            return 0
        if self._fighter_raid_defense_pressure(engine, platform) and current_total >= 5:
            return 3
        if current_total >= 4:
            return 2
        return 1

    def _fighter_air_salvo_count(self, engine, platform, weapon_id: str, target) -> int:
        if engine.inventory.get_weapon(weapon_id).max_range_km <= 60.0:
            return 1
        template = engine.inventory.get_platform(platform.template_id)
        initial_total = sum(
            count
            for candidate_id, count in template.weapon_loadout.items()
            if candidate_id in engine.inventory.weapons and engine.inventory.get_weapon(candidate_id).category == WeaponCategory.AIR_TO_AIR
        )
        current_total = self._remaining_category_ammo(platform, engine, WeaponCategory.AIR_TO_AIR)
        if platform.doctrine == EngagementDoctrine.FIRST_CONTACT and current_total > max(2, int(initial_total * 0.5)):
            return 2
        if target.platform_type == "awacs" and current_total > 1:
            return 2
        return 1

    def _fighter_air_reserve(self, engine, platform) -> int:
        template = engine.inventory.get_platform(platform.template_id)
        initial_total = sum(
            count
            for candidate_id, count in template.weapon_loadout.items()
            if candidate_id in engine.inventory.weapons and engine.inventory.get_weapon(candidate_id).category == WeaponCategory.AIR_TO_AIR
        )
        if initial_total <= 2:
            return 0
        return max(1, int(round(initial_total * 0.08)))

    def _fighter_should_spend_reserve(self, engine, platform, target) -> bool:
        if target.platform_type == "awacs":
            return True
        hostile_air = [
            item
            for item in engine.platforms.values()
            if item.alive and item.side != platform.side and item.platform_type in {"fighter", "awacs", "helicopter"}
        ]
        friendly_shooters = [
            item
            for item in engine.platforms.values()
            if item.alive
            and item.side == platform.side
            and item.platform_type == "fighter"
            and self._remaining_category_ammo(item, engine, WeaponCategory.AIR_TO_AIR) > 0
        ]
        if len(hostile_air) <= 1 or len(friendly_shooters) <= 1:
            return True
        if distance_km(platform.position, target.position) <= 28.0:
            return True
        if self._existing_air_commitments(engine, platform.side.value, target.entity_id) == 0 and len(hostile_air) <= len(friendly_shooters):
            return True
        return False

    def _existing_air_commitments(self, engine, side: str, target_id: str) -> int:
        return sum(
            1
            for projectile in engine.projectiles.values()
            if projectile.alive
            and projectile.side.value == side
            and projectile.category == WeaponCategory.AIR_TO_AIR
            and projectile.target_id == target_id
        )

    def _desired_air_commitments(self, target) -> int:
        if target.platform_type == "awacs":
            return 2
        if getattr(target, "scenario_role", "") == "fighter_strike":
            return 2
        if target.platform_type == "helicopter":
            return 1
        return 1

    def _engagement_factor(self, target) -> float:
        rcs = max(getattr(target, "radar_cross_section_m2", 1.0), 0.01)
        stealth = max(getattr(target, "stealth_factor", 1.0), 0.35)
        if getattr(target, "platform_type", "") in {"fighter", "awacs", "helicopter"}:
            factor = (rcs / 5.0) ** 0.05 / (stealth ** 0.12)
            return max(0.7, min(1.05, factor))
        if getattr(target, "platform_type", "") in {"destroyer", "cruiser", "frigate", "corvette", "carrier", "launcher"}:
            factor = (max(rcs, 40.0) / 500.0) ** 0.03 / (stealth ** 0.06)
            return max(0.86, min(1.08, factor))
        factor = (max(rcs, 1.0) / 15.0) ** 0.04 / (stealth ** 0.08)
        return max(0.8, min(1.05, factor))

    def _desired_air_defense_commitments(self, engine, target, contact) -> int:
        desired = self._desired_air_commitments(target)
        if target.platform_type == "awacs":
            desired = max(desired, 2)
        if contact.source_sensor.startswith("network:") and target.platform_type == "fighter":
            hostile_air = sum(
                1
                for item in engine.platforms.values()
                if item.alive and item.side != target.side and item.platform_type in {"fighter", "awacs", "helicopter"}
            )
            if hostile_air >= 8:
                desired = max(desired, 2)
        return min(2, desired)

    def _existing_surface_commitments(self, engine, side: str, target_id: str) -> int:
        return sum(
            1
            for projectile in engine.projectiles.values()
            if projectile.alive
            and projectile.side.value == side
            and not projectile.defensive
            and projectile.category in {WeaponCategory.ANTI_SHIP, WeaponCategory.TORPEDO}
            and projectile.target_id == target_id
        )

    def _desired_surface_commitments(self, target) -> int:
        if target.platform_type == "carrier":
            return 3
        if target.platform_type in {"cruiser", "destroyer", "submarine", "launcher"}:
            return 2
        return 1

    def _existing_defensive_commitments(self, engine, side: str, target_id: str) -> int:
        return sum(
            1
            for projectile in engine.projectiles.values()
            if projectile.alive
            and projectile.side.value == side
            and projectile.defensive
            and projectile.target_id == target_id
        )

    def _projectile_defense_budget(self, platform) -> int:
        if platform.platform_type == "fighter":
            return 2
        if platform.platform_type in {"destroyer", "cruiser"}:
            return 6
        if platform.platform_type in {"frigate", "carrier"}:
            return 4
        if platform.platform_type == "corvette":
            return 3
        return 2

    def _air_defense_budget(self, platform) -> int:
        if platform.platform_type == "fighter":
            return 2
        if platform.platform_type in {"destroyer", "cruiser"}:
            return 6
        if platform.platform_type in {"frigate", "carrier"}:
            return 4
        if platform.platform_type == "corvette":
            return 3
        return 2

    def _desired_projectile_commitments(self, engine, projectile) -> int:
        if projectile.metadata.get("flight_profile") == "ballistic":
            return 2
        if projectile.category == WeaponCategory.ANTI_SHIP:
            target = engine.platforms.get(projectile.target_id)
            if target is not None and target.platform_type in {"carrier", "cruiser", "destroyer"}:
                return 2 + min(1, int(projectile.metadata.get("failed_intercepts", 0)))
            return 2 + min(1, int(projectile.metadata.get("failed_intercepts", 0)))
        return 1

    def _fighter_raid_defense_pressure(self, engine, platform) -> bool:
        hostile_air_contacts = sum(
            1
            for contact in platform.contacts.values()
            if contact.target_id in engine.platforms
            and engine.platforms[contact.target_id].alive
            and engine.platforms[contact.target_id].side != platform.side
            and engine.platforms[contact.target_id].platform_type in {"fighter", "awacs", "helicopter"}
        )
        inbound_raids = sum(
            1
            for projectile in engine.projectiles.values()
            if projectile.alive
            and projectile.side != platform.side
            and projectile.category == WeaponCategory.ANTI_SHIP
            and projectile.target_id in engine.platforms
            and engine.platforms[projectile.target_id].side == platform.side
        )
        return hostile_air_contacts >= 3 or inbound_raids >= 2

    def _nearest_defense_threat_distance(self, engine, platform) -> float:
        distances: list[float] = []
        for projectile in engine.projectiles.values():
            if not projectile.alive or projectile.side == platform.side or projectile.defensive:
                continue
            if projectile.target_id == platform.entity_id:
                distances.append(distance_km(platform.position, projectile.position))
            elif projectile.target_id in engine.platforms and engine.platforms[projectile.target_id].side == platform.side:
                distances.append(distance_km(platform.position, projectile.position))
        for contact in platform.contacts.values():
            if contact.target_id not in engine.platforms:
                continue
            target = engine.platforms[contact.target_id]
            if target.alive and target.side != platform.side and target.platform_type in {"fighter", "awacs", "helicopter"}:
                distances.append(contact.distance_km)
        return min(distances, default=99999.0)

    def _defense_priority_rank(self, platform) -> int:
        return {
            "destroyer": 0,
            "cruiser": 1,
            "frigate": 2,
            "carrier": 3,
            "corvette": 4,
            "fighter": 5,
            "helicopter": 6,
            "awacs": 7,
            "submarine": 8,
        }.get(platform.platform_type, 9)

    def _guidance_label(self, engine, shooter, contact, target) -> str:
        if not str(contact.source_sensor).startswith("network:"):
            return ""
        for platform in engine.platforms.values():
            if platform.side != shooter.side or platform.entity_id == shooter.entity_id or not platform.alive:
                continue
            if target.entity_id not in platform.contacts:
                continue
            if platform.platform_type == "awacs":
                return "AWACS"
            if platform.platform_type == "fighter":
                return "FTR"
        return "ALLY"

    def _surface_guidance_label(self, engine, shooter, target) -> str:
        for platform in engine.platforms.values():
            if platform.side != shooter.side or platform.entity_id == shooter.entity_id or not platform.alive:
                continue
            if target.entity_id not in platform.contacts:
                continue
            if platform.platform_type == "awacs":
                return "AWACS"
            if platform.platform_type in {"fighter", "helicopter"}:
                return "FTR"
        return ""

    def _fighter_maneuver_rating(self, platform) -> float:
        table = {
            "rafale_m": 1.12,
            "su35s": 1.08,
            "su57_felon": 1.14,
            "f22_raptor": 1.18,
            "f35c": 1.0,
            "f35b": 0.96,
            "fa18e_super_hornet": 0.95,
            "f15ex_eagle_ii": 0.9,
            "mig29k": 1.0,
            "j35": 1.02,
            "j15_flying_shark": 0.84,
        }
        return table.get(platform.template_id, 0.9)

    def _dogfight_point(self, platform, target) -> tuple[float, float]:
        separation = max(0.1, distance_km(platform.position, target.position))
        pursuit = normalize(sub_vec(target.position, platform.position))
        if pursuit == (0.0, 0.0):
            pursuit = heading_to_vector(platform.heading_deg)
        if separation > 3.0:
            return target.position
        target_forward = heading_to_vector(target.heading_deg)
        lateral = (-pursuit[1], pursuit[0])
        rating_gap = self._fighter_maneuver_rating(platform) - self._fighter_maneuver_rating(target)
        offset_sign = 1.0 if hash(platform.entity_id) % 2 == 0 else -1.0
        lead = min(1.8, max(0.5, separation * 0.22))
        trail = min(0.9, max(0.2, separation * 0.12))
        lateral_offset = max(0.25, 0.65 + rating_gap * 0.35) * offset_sign
        return add_vec(
            target.position,
            add_vec(
                scale_vec(target_forward, lead),
                add_vec(scale_vec(pursuit, -trail), scale_vec(lateral, lateral_offset)),
            ),
        )

    def _close_air_merge(self, platform, target) -> bool:
        if target.platform_type not in {"fighter", "awacs", "helicopter"}:
            return False
        return distance_km(platform.position, target.position) <= 10.0

    def _should_force_dogfight(self, engine, platform, target) -> bool:
        del engine
        return self._close_air_merge(platform, target)

    def _bugout_point(self, platform, target) -> tuple[float, float]:
        retreat = normalize(sub_vec(platform.position, target.position))
        lateral = (-retreat[1], retreat[0])
        return add_vec(platform.position, add_vec(scale_vec(retreat, 42.0), scale_vec(lateral, 10.0)))

    def _intercept_course_point(self, platform, target) -> tuple[float, float]:
        lead_distance = min(24.0, max(8.0, distance_km(platform.position, target.position) * 0.12))
        return add_vec(target.position, scale_vec(heading_to_vector(target.heading_deg), lead_distance))

    def _guidance_track_point(self, platform, target) -> tuple[float, float]:
        direction = normalize(sub_vec(target.position, platform.position))
        if direction == (0.0, 0.0):
            direction = heading_to_vector(platform.heading_deg)
        return add_vec(target.position, scale_vec(direction, -14.0))

    def _projectile_intercept_point(self, projectile) -> tuple[float, float]:
        return add_vec(projectile.position, scale_vec(heading_to_vector(projectile.heading_deg), 3.0))

    def _should_flee_dogfight(self, engine, platform, target) -> bool:
        maneuver_gap = self._fighter_maneuver_rating(target) - self._fighter_maneuver_rating(platform)
        if maneuver_gap < 0.18:
            return False
        separation = distance_km(platform.position, target.position)
        if separation <= 10.0:
            return False
        if separation > 14.0:
            return False
        return chance(engine.rng, min(0.18, 0.06 + maneuver_gap * 0.3))

    def _gun_attack(self, engine, attacker, target) -> bool:
        if target.platform_type not in {"fighter", "awacs", "helicopter"}:
            return False
        separation = distance_km(attacker.position, target.position)
        if separation > 2.1 or not self._within_attack_cone(attacker, target, quality=ContactQuality.FIRE_CONTROL):
            return False
        attack_rating = self._fighter_maneuver_rating(attacker)
        defense_rating = max(0.7, self._fighter_maneuver_rating(target))
        hit_probability = min(0.42, max(0.12, 0.23 * (attack_rating / defense_rating) * (1.8 - separation / 2.1)))
        if chance(engine.rng, hit_probability):
            damage = 11.0 * attack_rating
            target.apply_damage(damage, 0.0)
            engine.spawn_effect("explosion", target.position, ttl=1.0)
            engine.logger.log("gun_hit", time_s=engine.current_time_s, attacker_id=attacker.entity_id, target_id=target.entity_id, damage=damage)
        else:
            engine.spawn_effect("miss", target.position, ttl=0.8)
            engine.logger.log("gun_burst_miss", time_s=engine.current_time_s, attacker_id=attacker.entity_id, target_id=target.entity_id)
        return True

    def _has_network_cue(self, platform) -> bool:
        return any(contact.source_sensor.startswith("network:") for contact in platform.contacts.values())

    def _pick_air_target(self, engine, platform):
        solution = self._pick_air_engagement(engine, platform)
        if solution is None:
            if platform.platform_type == "fighter":
                return self._pick_visual_air_target(engine, platform)
            return None
        return solution[1]

    def _pick_visual_air_target(self, engine, platform, max_distance_km: float = 12.0):
        if platform.platform_type != "fighter":
            return None
        candidates = [
            target
            for target in engine.platforms.values()
            if target.alive
            and target.side != platform.side
            and target.platform_type in {"fighter", "awacs", "helicopter"}
            and distance_km(platform.position, target.position) <= max_distance_km
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda target: (
                target.platform_type != "fighter",
                distance_km(platform.position, target.position),
            )
        )
        return candidates[0]

    def _fighter_missile_defense_target(self, engine, platform):
        if platform.platform_type != "fighter":
            return None
        if self._pick_air_target(engine, platform) is not None:
            return None
        clutter = engine.config.clutter_factor * engine.scenario_context.get("weather_factor", 1.0)
        best = None
        for projectile in engine.projectiles.values():
            if not projectile.alive or projectile.side == platform.side or projectile.defensive:
                continue
            if projectile.category != WeaponCategory.ANTI_SHIP:
                continue
            if projectile.target_id not in engine.platforms:
                continue
            defended = engine.platforms[projectile.target_id]
            if not defended.alive or defended.side != platform.side or defended.platform_type in {"fighter", "awacs", "helicopter"}:
                continue
            local_window = engine.detector.projectile_detection_range_km(platform, projectile, clutter)
            network_window, _source = engine.detector.network_projectile_cue(platform, list(engine.platforms.values()), projectile, clutter)
            separation = distance_km(platform.position, projectile.position)
            if separation > max(local_window, network_window, 0.0) + 12.0:
                continue
            score = (
                defended.platform_type != "carrier",
                distance_km(projectile.position, defended.position),
                separation,
            )
            if best is None or score < best[0]:
                best = (score, projectile)
        if best is None:
            return None
        return best[1]

    def _pick_ship_air_defense_target(self, engine, platform):
        solutions = self._ship_air_defense_targets(engine, platform)
        if not solutions:
            return None
        return solutions[0]

    def _ship_air_defense_targets(self, engine, platform):
        if platform.platform_type in {"fighter", "awacs", "helicopter", "submarine"}:
            return []
        solutions = []
        for contact in platform.contacts.values():
            if contact.target_id not in engine.platforms:
                continue
            target = engine.platforms[contact.target_id]
            if not target.alive or target.platform_type not in {"fighter", "awacs", "helicopter"}:
                continue
            if contact.quality not in {ContactQuality.TRACK, ContactQuality.FIRE_CONTROL} and not (
                contact.quality == ContactQuality.SEARCH and contact.source_sensor.startswith("network:")
            ):
                continue
            quality_rank = {
                ContactQuality.SEARCH: 1,
                ContactQuality.TRACK: 2,
                ContactQuality.FIRE_CONTROL: 3,
            }[contact.quality]
            solutions.append((target.platform_type != "fighter", -quality_rank, contact.distance_km, contact, target))
        if not solutions:
            return []
        solutions.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(item[3], item[4]) for item in solutions]

    def _pick_air_engagement(self, engine, platform):
        air_contacts = self._air_engagement_solutions(engine, platform)
        if not air_contacts:
            return None
        return air_contacts[0]

    def _air_engagement_solutions(self, engine, platform):
        air_contacts = []
        for contact in platform.contacts.values():
            if contact.target_id not in engine.platforms:
                continue
            target = engine.platforms[contact.target_id]
            if not target.alive or target.platform_type not in {"fighter", "awacs", "helicopter"}:
                continue
            if contact.quality not in {ContactQuality.TRACK, ContactQuality.FIRE_CONTROL} and not (
                contact.quality == ContactQuality.SEARCH and contact.source_sensor.startswith("network:")
            ):
                continue
            quality_rank = {
                ContactQuality.SEARCH: 1,
                ContactQuality.TRACK: 2,
                ContactQuality.FIRE_CONTROL: 3,
            }[contact.quality]
            air_contacts.append(
                (
                    target.platform_type != "fighter",
                    -quality_rank,
                    contact.distance_km,
                    contact,
                    target,
                )
            )
        if not air_contacts:
            return []
        air_contacts.sort(key=lambda item: (item[0], item[1], item[2]))
        return [(item[3], item[4]) for item in air_contacts]

    def _within_attack_cone(self, platform, target, allow_offboard: bool = False, quality: ContactQuality | None = None) -> bool:
        if allow_offboard:
            return True
        direction = normalize(sub_vec(target.position, platform.position))
        forward = normalize((math.sin(math.radians(platform.heading_deg)), -math.cos(math.radians(platform.heading_deg))))
        dot = forward[0] * direction[0] + forward[1] * direction[1]
        if quality == ContactQuality.FIRE_CONTROL:
            return dot >= -0.25
        if quality == ContactQuality.TRACK:
            return dot >= -0.1
        return dot >= 0.15

    def _coordinate_forces(self, engine) -> None:
        for side in {platform.side for platform in engine.platforms.values()}:
            formation = [platform for platform in engine.platforms.values() if platform.alive and platform.side == side]
            if not formation:
                continue
            leader = self._pick_leader(formation)
            contacts = self._best_network_targets(formation, engine)
            committed_surface_ids = self._committed_surface_ids(formation, contacts[0]) if contacts else set()
            air_support_targets = self._formation_air_support_targets(formation, engine)
            for index, platform in enumerate(sorted(formation, key=lambda item: item.entity_id)):
                state = self._platform_state(platform.entity_id)
                incoming = self._incoming_threat(engine, platform) if platform.platform_type == "fighter" else None
                air_threat = self._pick_air_target(engine, platform) if platform.platform_type == "fighter" else None
                forced_close_merge = platform.platform_type == "fighter" and air_threat is not None and self._should_force_dogfight(engine, platform, air_threat)
                if platform.manual_waypoints and platform.waypoints:
                    if incoming is not None:
                        platform.air_state = "evading"
                        state["evade_until_s"] = engine.current_time_s + 8.0
                        self._set_waypoint(platform, self._evasion_point(platform, incoming), state, engine.current_time_s, immediate=True)
                    elif forced_close_merge:
                        platform.air_state = "dogfight"
                        self._set_waypoint(platform, self._dogfight_point(platform, air_threat), state, engine.current_time_s, immediate=True)
                    continue
                if platform.platform_type not in {"fighter", "awacs", "helicopter", "submarine"} and platform.waypoints:
                    continue
                if engine.current_time_s < float(state.get("next_nav_update_s", 0.0)):
                    continue
                state["next_nav_update_s"] = engine.current_time_s + self._nav_update_interval(platform, len(formation))
                if platform.platform_type == "carrier":
                    self._set_waypoint(platform, self._carrier_axis_point(platform, contacts), state, engine.current_time_s)
                elif platform.platform_type == "awacs":
                    if platform.scenario_role == "awacs_orbit" and platform.assigned_guard_target_id in engine.platforms and not contacts:
                        self._set_patrol_waypoints(platform, engine._awacs_barrier_pattern(engine.platforms[platform.assigned_guard_target_id]), state)
                        continue
                    anchor = leader if leader.platform_type == "carrier" else platform
                    if air_support_targets:
                        focus = min(air_support_targets, key=lambda target: distance_km(anchor.position, target.position))
                        self._set_waypoint(platform, self._standoff_point(anchor.position, focus.position, 110.0), state, engine.current_time_s)
                    elif contacts:
                        self._set_waypoint(platform, self._standoff_point(anchor.position, contacts[0].position, 130.0), state, engine.current_time_s)
                    else:
                        self._set_waypoint(platform, (anchor.position[0] + 110.0, anchor.position[1] - 80.0), state, engine.current_time_s)
                elif platform.platform_type == "fighter":
                    if incoming is not None:
                        platform.air_state = "evading"
                        state["evade_until_s"] = engine.current_time_s + 8.0
                        self._set_waypoint(platform, self._evasion_point(platform, incoming), state, engine.current_time_s, immediate=True)
                        continue
                    missile_defense_target = self._fighter_missile_defense_target(engine, platform)
                    if missile_defense_target is not None:
                        platform.air_state = "missile_defense"
                        self._set_waypoint(platform, self._projectile_intercept_point(missile_defense_target), state, engine.current_time_s)
                        continue
                    support_target = self._support_target(engine, platform, state, key="guided_target_id")
                    if support_target is not None:
                        platform.air_state = "guiding"
                        self._set_waypoint(platform, self._guidance_track_point(platform, support_target), state, engine.current_time_s)
                        continue
                    if air_threat is not None:
                        if forced_close_merge or self._remaining_category_ammo(platform, engine, WeaponCategory.AIR_TO_AIR) <= 0:
                            if self._should_flee_dogfight(engine, platform, air_threat):
                                platform.air_state = "disengaging"
                                self._set_waypoint(platform, self._bugout_point(platform, air_threat), state, engine.current_time_s)
                                continue
                            platform.air_state = "dogfight"
                            self._set_waypoint(platform, self._dogfight_point(platform, air_threat), state, engine.current_time_s)
                        else:
                            platform.air_state = "intercept"
                            intercept_point = self._intercept_course_point(platform, air_threat)
                            if distance_km(platform.position, air_threat.position) <= 22.0:
                                intercept_point = air_threat.position
                            self._set_waypoint(platform, intercept_point, state, engine.current_time_s)
                        continue
                    if air_support_targets:
                        support = min(air_support_targets, key=lambda target: distance_km(platform.position, target.position))
                        if distance_km(platform.position, support.position) <= 220.0:
                            platform.air_state = "supporting"
                            self._set_waypoint(platform, self._intercept_course_point(platform, support), state, engine.current_time_s)
                            continue
                    if platform.scenario_role == "fighter_patrol" and platform.assigned_guard_target_id in engine.platforms and not contacts:
                        platform.air_state = "on_station"
                        self._set_patrol_waypoints(platform, engine._fighter_search_pattern(engine.platforms[platform.assigned_guard_target_id]), state)
                        continue
                    surface_target = self._pick_surface_target(engine, platform)
                    anti_ship_weapon = self._pick_weapon(platform, engine, WeaponCategory.ANTI_SHIP)
                    if surface_target is not None and anti_ship_weapon is not None:
                        weapon_range = engine.inventory.get_weapon(anti_ship_weapon).max_range_km
                        platform.air_state = "strike_ingress"
                        self._set_waypoint(platform, self._standoff_point(platform.position, surface_target.position, max(18.0, weapon_range * 0.58)), state, engine.current_time_s)
                    elif platform.scenario_role == "fighter_strike" and contacts:
                        platform.air_state = "strike_ingress"
                        self._set_waypoint(platform, self._standoff_point(platform.position, contacts[0].position, 90.0), state, engine.current_time_s)
                    elif contacts:
                        target = contacts[0]
                        platform.air_state = "on_station"
                        self._set_waypoint(platform, self._standoff_point(platform.position, target.position, 60.0), state, engine.current_time_s)
                    else:
                        platform.air_state = "on_station"
                        if leader.platform_type == "carrier":
                            self._set_waypoint(platform, (leader.position[0] + 60.0, leader.position[1] + (-30.0 if index % 2 == 0 else 30.0)), state, engine.current_time_s)
                        else:
                            self._set_waypoint(platform, self._forward_heading_point(platform, 90.0), state, engine.current_time_s)
                elif platform.platform_type == "helicopter":
                    if platform.scenario_role == "helicopter_patrol" and platform.assigned_guard_target_id in engine.platforms and not contacts:
                        self._set_patrol_waypoints(platform, engine._helicopter_search_pattern(engine.platforms[platform.assigned_guard_target_id]), state)
                        continue
                    anchor = leader if leader.platform_type == "carrier" else platform
                    self._set_waypoint(platform, (anchor.position[0] + 24.0, anchor.position[1] + 28.0), state, engine.current_time_s)
                elif platform.platform_type == "submarine":
                    surface_target = self._pick_surface_target(engine, platform)
                    if surface_target is not None:
                        torpedo = self._pick_weapon(platform, engine, WeaponCategory.TORPEDO)
                        if torpedo is not None and distance_km(platform.position, surface_target.position) <= engine.inventory.get_weapon(torpedo).max_range_km * 0.92:
                            self._set_waypoint(platform, self._standoff_point(platform.position, surface_target.position, 10.0), state, engine.current_time_s)
                        else:
                            self._set_waypoint(platform, self._standoff_point(platform.position, surface_target.position, 42.0), state, engine.current_time_s)
                    elif leader.platform_type == "carrier":
                        escort_offset = self._escort_offset(index + 2)
                        self._set_waypoint(platform, add_vec(leader.position, (escort_offset[0] - 18.0, escort_offset[1] * 1.3)), state, engine.current_time_s)
                elif platform.scenario_role == "fighter_strike" and contacts:
                    self._set_waypoint(platform, self._standoff_point(platform.position, contacts[0].position, 90.0), state, engine.current_time_s)
                else:
                    if leader.platform_type == "carrier" and platform.entity_id != leader.entity_id:
                        platform.assigned_guard_target_id = leader.entity_id
                        escort_offset = self._escort_offset(index)
                        anchor = add_vec(leader.position, escort_offset)
                        if contacts and platform.entity_id in committed_surface_ids:
                            threat_axis = normalize(sub_vec(contacts[0].position, leader.position))
                            anchor = add_vec(anchor, scale_vec(threat_axis, 18.0))
                        self._set_waypoint(platform, anchor, state, engine.current_time_s)
                        if platform.doctrine == EngagementDoctrine.FIRST_CONTACT:
                            platform.doctrine = EngagementDoctrine.DEFEND_CARRIER
                    elif contacts:
                        self._set_waypoint(platform, self._standoff_point(platform.position, contacts[0].position, 120.0), state, engine.current_time_s)

    def _pick_leader(self, formation):
        carriers = [platform for platform in formation if platform.platform_type == "carrier"]
        if carriers:
            return sorted(carriers, key=lambda item: item.max_hull_points, reverse=True)[0]
        return sorted(formation, key=lambda item: item.max_hull_points, reverse=True)[0]

    def _best_network_targets(self, formation, engine):
        targets = {}
        for platform in formation:
            for contact in platform.contacts.values():
                if (
                    contact.target_id in engine.platforms
                    and engine.platforms[contact.target_id].alive
                    and engine.platforms[contact.target_id].platform_type not in {"fighter", "awacs", "helicopter"}
                ):
                    target = engine.platforms[contact.target_id]
                    existing = targets.get(target.entity_id)
                    if existing is None or self._target_priority(target) > self._target_priority(existing):
                        targets[target.entity_id] = target
        return sorted(targets.values(), key=self._target_priority, reverse=True)

    def _target_priority(self, target) -> float:
        bonus = {
            "carrier": 220.0,
            "cruiser": 160.0,
            "destroyer": 140.0,
            "frigate": 110.0,
            "submarine": 135.0,
            "launcher": 125.0,
            "corvette": 80.0,
        }.get(target.platform_type, 60.0)
        return target.hull_points + bonus

    def _committed_surface_ids(self, formation, target) -> set[str]:
        combatants = [
            platform
            for platform in formation
            if platform.platform_type not in {"fighter", "awacs", "helicopter"}
        ]
        if not combatants:
            return set()
        commit_count = min(
            len(combatants),
            max(1, len(combatants) // 4)
            + (3 if target.platform_type == "carrier" else 2 if target.platform_type in {"cruiser", "destroyer", "submarine"} else 1),
        )
        ranked = sorted(combatants, key=lambda platform: distance_km(platform.position, target.position))
        return {platform.entity_id for platform in ranked[:commit_count]}

    def _formation_air_support_targets(self, formation, engine):
        targets = []
        seen: set[str] = set()
        for platform in formation:
            for contact in platform.contacts.values():
                if contact.target_id not in engine.platforms or contact.target_id in seen:
                    continue
                target = engine.platforms[contact.target_id]
                if target.alive and target.platform_type in {"fighter", "awacs", "helicopter"}:
                    targets.append(target)
                    seen.add(target.entity_id)
        return targets

    def _pick_surface_target(self, engine, platform):
        valid_contacts = [
            (contact.distance_km, engine.platforms[contact.target_id])
            for contact in platform.contacts.values()
            if contact.quality in self._allowed_contact_qualities(platform)
            and contact.target_id in engine.platforms
            and engine.platforms[contact.target_id].alive
            and engine.platforms[contact.target_id].platform_type not in {"fighter", "awacs", "helicopter"}
        ]
        if not valid_contacts:
            return None
        valid_contacts.sort(key=lambda item: item[0])
        return valid_contacts[0][1]

    def _incoming_threat(self, engine, platform):
        incoming = [
            projectile
            for projectile in engine.projectiles.values()
            if projectile.alive and projectile.side != platform.side and projectile.target_id == platform.entity_id and projectile.category == WeaponCategory.AIR_TO_AIR
        ]
        if not incoming:
            return None
        incoming.sort(key=lambda projectile: distance_km(platform.position, projectile.position))
        nearest = incoming[0]
        if distance_km(platform.position, nearest.position) > 32.0:
            return None
        return nearest

    def _evasion_point(self, platform, projectile) -> tuple[float, float]:
        incoming = normalize(sub_vec(platform.position, projectile.position))
        lateral = (-incoming[1], incoming[0])
        return add_vec(platform.position, add_vec(scale_vec(incoming, 18.0), scale_vec(lateral, 12.0)))

    def _support_target(self, engine, platform, state, key: str):
        target_id = state.get(key)
        if not target_id or state.get("support_until_s", 0.0) < engine.current_time_s:
            return None
        target = engine.platforms.get(str(target_id))
        if target is None or not target.alive:
            return None
        return target

    def _escort_offset(self, index: int) -> tuple[float, float]:
        pattern = [(-20.0, -35.0), (-20.0, 35.0), (-50.0, -60.0), (-50.0, 60.0), (-85.0, 0.0)]
        return pattern[index % len(pattern)]

    def _carrier_axis_point(self, carrier, contacts):
        if contacts:
            return self._standoff_point(carrier.position, contacts[0].position, 220.0)
        return (carrier.position[0] + 22.0, carrier.position[1])

    def _forward_heading_point(self, platform, distance_km_ahead: float) -> tuple[float, float]:
        forward = heading_to_vector(platform.heading_deg)
        return add_vec(platform.position, scale_vec(forward, distance_km_ahead))

    def _standoff_point(self, current: tuple[float, float], target: tuple[float, float], standoff_km: float) -> tuple[float, float]:
        direction_from_target = normalize(sub_vec(current, target))
        return add_vec(target, scale_vec(direction_from_target, standoff_km))

    def _nav_update_interval(self, platform, formation_size: int) -> float:
        if formation_size >= 160:
            if platform.platform_type == "fighter":
                return 3.0
            if platform.platform_type in {"awacs", "helicopter"}:
                return 3.5
            return 4.0
        if formation_size >= 100:
            if platform.platform_type == "fighter":
                return 2.5
            if platform.platform_type in {"awacs", "helicopter"}:
                return 3.0
            return 3.5
        if formation_size >= 50:
            if platform.platform_type == "fighter":
                return 2.0
            if platform.platform_type in {"awacs", "helicopter"}:
                return 2.0
            return 3.0
        if platform.platform_type == "fighter":
            return 1.0
        return 2.0

    def _set_patrol_waypoints(self, platform, waypoints: list[tuple[float, float]], state) -> None:
        if platform.manual_waypoints:
            return
        normalized = [tuple(waypoint) for waypoint in waypoints]
        if platform.waypoints == normalized:
            return
        platform.waypoints = normalized
        state["last_waypoint"] = normalized[0] if normalized else None

    def _set_waypoint(self, platform, waypoint: tuple[float, float], state, current_time_s: float, immediate: bool = False) -> None:
        if platform.manual_waypoints and not immediate:
            return
        current = platform.waypoints[0] if platform.waypoints else None
        previous = state.get("last_waypoint")
        if not immediate and current is not None and distance_km(current, waypoint) < 10.0:
            return
        if not immediate and previous is not None and distance_km(previous, waypoint) < 10.0 and platform.waypoints:
            return
        platform.waypoints = [waypoint]
        state["last_waypoint"] = waypoint
        if immediate:
            state["next_nav_update_s"] = current_time_s + 1.0


if __name__ == "__main__":
    ai = AIController()
    assert ai._platform_state("demo")["next_salvo_time_s"] == 0.0
    print("ai_controller smoke test ok")
