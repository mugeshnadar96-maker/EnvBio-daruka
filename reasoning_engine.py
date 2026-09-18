"""
reasoning_engine.py
--------------------
Deterministic, evidence-backed multi-metric reasoning module. Each Rule
requires >=3 environmental variables (spanning soil/land/climate/biodiversity/
human-impact categories) to fire, and every recommendation carries the
mandatory six output fields: what_to_do, impacted_metrics, scientific_reasoning,
evidence, time_horizon, confidence.

This module intentionally does NOT call an LLM. It is the "hardcoded
reasoning" layer required by the spec, kept separate from (and combined
with) the RAG retrieval layer in conversation_agent.py.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class Recommendation:
    what_to_do: str
    impacted_metrics: List[str]
    scientific_reasoning: str
    evidence: str
    time_horizon: str
    confidence: str
    rule_id: str


@dataclass
class SiteState:
    """Normalized input variables. Any field may be None if unknown."""
    soc_percent: Optional[float] = None          # soil organic carbon %
    ph: Optional[float] = None
    moisture: Optional[str] = None                # low/medium/high
    bulk_density: Optional[float] = None           # g/cm3
    rainfall_mm: Optional[float] = None
    rainfall_category: Optional[str] = None        # low/medium/high
    mean_temp_c: Optional[float] = None
    aridity_index: Optional[float] = None
    land_use: Optional[str] = None                 # monoculture / mixed / agroforestry / fallow / pasture
    region: Optional[str] = None
    fragmentation_index: Optional[float] = None     # 0-1
    tillage_intensity: Optional[str] = None         # low/medium/high
    fertilizer_use: Optional[str] = None            # low/medium/high
    deforestation_adjacent: Optional[bool] = None
    species_richness_index: Optional[float] = None  # 0-1
    habitat_diversity_index: Optional[float] = None  # 0-1
    pollinator_abundance: Optional[str] = None        # low/medium/high
    grazing_type: Optional[str] = None                 # continuous/rotational
    biodiversity_habitat_index: Optional[float] = None  # 0-1, from geo-lookup (India, CSIRO BHI v4)
    bhi_trend: Optional[float] = None                   # 2000-2024 change, from geo-lookup

    def rainfall_bucket(self) -> Optional[str]:
        if self.rainfall_category:
            return self.rainfall_category
        if self.rainfall_mm is not None:
            if self.rainfall_mm < 500:
                return "low"
            if self.rainfall_mm < 1000:
                return "medium"
            return "high"
        return None

    def known_fields(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# RULE DEFINITIONS
# Each rule: id, required_vars (for completeness checking), condition(state)->bool,
# and a builder(state) -> Recommendation
# ---------------------------------------------------------------------------

def _rule_1(s: SiteState) -> bool:
    return (s.soc_percent is not None and s.soc_percent < 0.5
            and s.rainfall_bucket() == "low"
            and s.land_use in ("monoculture", "monoculture_wheat", "monoculture_cereal"))

def _build_1(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Transition to agroforestry with legume-based intercropping "
                    "(e.g., wheat + pigeon pea + Faidherbia albida / Acacia / Leucaena)",
        impacted_metrics=[
            "Soil organic carbon (+15-25% over 2-3 yrs)",
            "Habitat diversity index (+30%)",
            "Water retention / plant-available moisture (+10-15%)",
        ],
        scientific_reasoning=(
            "Legume trees and cover crops fix atmospheric N2 via rhizobial symbiosis, raising "
            "microbial biomass and root-derived carbon inputs. Tree canopies reduce soil surface "
            "temperature and evaporative water loss, and deep roots access subsoil moisture that "
            "shallow-rooted monocultures cannot reach, buffering low-rainfall stress."
        ),
        evidence="FAO 'Recarbonizing Global Soils' 2021: 73 comparisons across 18 studies, "
                 "+15% SOC from intercropping [17]; IPCC AR6 WGIII Ch.7: agroforestry sequesters "
                 "0.3-15 tC/ha/yr in aboveground biomass [43]",
        time_horizon="Medium (2-4 yrs for measurable SOC gain)",
        confidence="High",
        rule_id="R1_low_soc_low_rain_monoculture",
    )


def _rule_2(s: SiteState) -> bool:
    return (s.species_richness_index is not None and s.species_richness_index < 0.3
            and s.fragmentation_index is not None and s.fragmentation_index > 0.6
            and s.land_use is not None)

def _build_2(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Establish linear habitat corridors with native drought-resistant shrubs "
                    "(e.g., Prosopis, Acacia) connecting fragmented patches",
        impacted_metrics=[
            "Species richness (+20-40% within 3 yrs)",
            "Habitat connectivity (fragmentation index reduction)",
            "Pollinator abundance",
        ],
        scientific_reasoning=(
            "Corridors reduce the effective isolation distance between habitat patches, enabling "
            "gene flow and recolonization for area-sensitive species. Native shrub cover restores "
            "nectar/pollen and nesting resources along field margins, reducing edge effects."
        ),
        evidence="OECD 2023 Farmland Habitat Biodiversity Indicator Guidelines: corridors increase "
                 "farmland species richness by 20-40% within 3 years [18]",
        time_horizon="Medium (2-4 yrs)",
        confidence="Medium",
        rule_id="R2_low_richness_high_fragmentation",
    )


def _rule_3(s: SiteState) -> bool:
    return (s.tillage_intensity == "high" and s.soc_percent is not None and s.soc_percent < 1.0
            and s.land_use is not None)

def _build_3(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Shift from conventional intensive tillage to reduced/no-till management",
        impacted_metrics=[
            "Soil organic carbon (+0.2-0.5 tC/ha/yr topsoil accumulation)",
            "Bulk density (-0.1-0.2 g/cm3 over 5 yrs)",
            "Water infiltration",
        ],
        scientific_reasoning=(
            "Tillage physically disrupts soil aggregates, exposing protected organic matter to "
            "microbial oxidation and accelerating carbon loss. Reduced tillage preserves aggregate "
            "structure and fungal hyphal networks, slowing mineralization and improving pore "
            "continuity for infiltration."
        ),
        evidence="FAO 'Recarbonizing Global Soils' 2021: reduced tillage +0.2-0.5 tC/ha/yr topsoil "
                 "SOC accumulation over 5-10 yrs [17]; SoilHealthDB-V2: bulk density reduction "
                 "-0.1-0.2 g/cm3 under sustained organic/reduced-till management [32]",
        time_horizon="Long (>3 yrs for full structural recovery)",
        confidence="High",
        rule_id="R3_high_tillage_low_soc",
    )


def _rule_4(s: SiteState) -> bool:
    return (s.land_use == "fallow" or s.moisture == "low") and s.rainfall_bucket() in ("low", "medium") and s.soc_percent is not None

def _build_4(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Replace bare fallow periods with cover cropping (legume or mixed-species cover)",
        impacted_metrics=[
            "Soil moisture retention (+10-15% plant-available water)",
            "Soil organic carbon",
            "Microbial biomass diversity (+20-35%)",
        ],
        scientific_reasoning=(
            "Living roots during fallow periods maintain rhizosphere carbon exudation and microbial "
            "activity that bare soil loses. Residue cover reduces evaporative loss and raindrop "
            "impact erosion, while root channels improve water infiltration and storage."
        ),
        evidence="SoilHealthDB-V2 (Nature Scientific Data): cover cropping +10-15% plant-available "
                 "water capacity [3]; diversified rotations +20-35% microbial biomass carbon [34]",
        time_horizon="Short to Medium (<1-2 yrs for moisture effect, 2-3 yrs for SOC)",
        confidence="Medium",
        rule_id="R4_fallow_or_low_moisture",
    )


def _rule_5(s: SiteState) -> bool:
    return (s.fertilizer_use == "high" and s.land_use in ("monoculture", "monoculture_wheat", "monoculture_cereal")
            and (s.pollinator_abundance == "low" or (s.species_richness_index is not None and s.species_richness_index < 0.4)))

def _build_5(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Reduce synthetic nitrogen fertilizer intensity and integrate sown flower strips "
                    "/ uncultivated field margins",
        impacted_metrics=[
            "Pollinator abundance (+25-45% near flower strips within 2 yrs)",
            "Soil microbial diversity (avoids 10-20% suppression from N overuse)",
            "Species richness",
        ],
        scientific_reasoning=(
            "Excess synthetic N alters soil microbial community composition, favoring fast-cycling "
            "taxa over diverse fungal networks, and nitrate runoff can degrade adjacent habitat "
            "quality. Flower strips restore forage continuity across the season for wild pollinators "
            "displaced by input-intensive monocultures."
        ),
        evidence="OECD 2023 Farmland Habitat Biodiversity Indicator Guidelines: high-input systems "
                 "show -30-50% pollinator abundance vs low-input; flower strips +25-45% pollinator "
                 "abundance within 2 yrs [18]; SoilHealthDB-V2: N overuse linked to 10-20% microbial "
                 "diversity reduction [3]",
        time_horizon="Short to Medium (<1-2 yrs for pollinator response)",
        confidence="Medium",
        rule_id="R5_high_fertilizer_low_pollinators",
    )


def _rule_6(s: SiteState) -> bool:
    return (s.land_use in ("monoculture", "monoculture_wheat", "monoculture_cereal")
            and s.habitat_diversity_index is not None and s.habitat_diversity_index < 0.4
            and s.rainfall_bucket() is not None)

def _build_6(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Diversify cropping structure into polyculture / multi-strata agroforestry "
                    "(herb + shrub + tree layers)",
        impacted_metrics=[
            "Habitat diversity index (+30% vs monoculture)",
            "Species richness across trophic guilds",
            "Microclimate buffering (soil temperature, evapotranspiration)",
        ],
        scientific_reasoning=(
            "Vertical and species structural diversity creates more ecological niches, supporting "
            "a wider range of trophic guilds (pollinators, predators, decomposers) than a single-"
            "canopy monoculture. Multiple canopy layers also buffer understory microclimate."
        ),
        evidence="FAO 'Recarbonizing Global Soils' 2021 / IPCC AR6 WGIII Ch.7: +30% habitat "
                 "diversity index in agroforestry vs monoculture systems [17][43]",
        time_horizon="Medium (2-4 yrs)",
        confidence="Medium",
        rule_id="R6_monoculture_low_habitat_diversity",
    )


def _rule_7(s: SiteState) -> bool:
    return (s.deforestation_adjacent is True and s.species_richness_index is not None)

def _build_7(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Establish a buffer/riparian forest strip along the deforestation edge and "
                    "restrict further clearing within it",
        impacted_metrics=[
            "Species richness (edge effect mitigation)",
            "Pollinator and predator insect diversity",
            "Soil erosion / water quality",
        ],
        scientific_reasoning=(
            "Species richness declines sharply with distance from intact forest edge as forest-"
            "dependent taxa lose canopy cover and microclimate stability. A buffer strip restores "
            "partial edge habitat and slows further richness decline from ongoing clearing."
        ),
        evidence="IPCC AR6 WGIII Ch.7: species richness declines ~50% beyond 1km from intact forest "
                 "edge in deforestation-adjacent cropland [43]",
        time_horizon="Long (>3 yrs for community recovery)",
        confidence="Medium",
        rule_id="R7_deforestation_adjacent",
    )


def _rule_8(s: SiteState) -> bool:
    return (s.land_use in ("monoculture", "monoculture_wheat", "monoculture_cereal", "fallow")
            and s.fragmentation_index is not None and s.fragmentation_index > 0.5
            and s.rainfall_bucket() == "low")

def _build_8(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Combine drought-tolerant hedgerow corridors with contour-aligned planting to "
                    "double as windbreaks and habitat connectors",
        impacted_metrics=[
            "Fragmentation index (reduced isolation)",
            "Species richness (+20-40% within 3 yrs)",
            "Wind erosion / moisture retention",
        ],
        scientific_reasoning=(
            "In low-rainfall, fragmented landscapes, hedgerows serve a dual function: reducing wind "
            "speed at the soil surface (lowering evaporative moisture loss) while providing the "
            "structural connectivity that area-sensitive species need to move between patches."
        ),
        evidence="OECD 2023 Farmland Habitat Biodiversity Indicator Guidelines [18]; IPCC AR6 WGIII "
                 "Ch.7 on tree-shaded soil microclimate buffering, 2-4°C lower peak soil temperature [43]",
        time_horizon="Medium (2-4 yrs)",
        confidence="Medium",
        rule_id="R8_fragmented_low_rain",
    )


def _rule_9(s: SiteState) -> bool:
    return (s.land_use == "pasture" or s.land_use == "grassland") and s.soc_percent is not None and s.land_use is not None

def _build_9(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Maintain/extend permanent pasture rather than converting to arable, and "
                    "introduce rotational (not continuous) grazing",
        impacted_metrics=[
            "Soil organic carbon (grassland SOC ~1.5-2x higher than intensive arable)",
            "Species richness",
            "Bulk density / compaction risk (managed via rotation)",
        ],
        scientific_reasoning=(
            "Permanent grass cover maintains continuous root carbon input without the periodic "
            "disturbance of tillage, sustaining higher SOC stocks. Rotational grazing prevents the "
            "compaction and overgrazing that would otherwise offset this advantage."
        ),
        evidence="LUCAS Soil Survey (Eurostat) 2022: grassland/pasture SOC ~1.5-2x higher than "
                 "intensive arable on comparable soils [39]",
        time_horizon="Long (>3 yrs)",
        confidence="Medium",
        rule_id="R9_pasture_conservation",
    )


def _rule_10(s: SiteState) -> bool:
    return (s.aridity_index is not None and s.aridity_index < 0.2) and s.land_use is not None and s.soc_percent is not None

def _build_10(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Prioritize drought-adapted perennial N-fixing trees (Faidherbia albida is "
                    "particularly suited to arid zones as it sheds leaves in the wet season)",
        impacted_metrics=[
            "Soil organic carbon",
            "Water retention",
            "Microclimate (soil temperature buffering)",
        ],
        scientific_reasoning=(
            "Under high aridity, tree species selection matters: reverse-phenology species like "
            "Faidherbia albida avoid competing with crops for water during the growing season while "
            "still contributing deep-root carbon and canopy shading in the dry season."
        ),
        evidence="IPCC AR6 WGIII Ch.7: agroforestry carbon sequestration range 0.3-15 tC/ha/yr, with "
                 "species selection strongly affecting arid-zone performance [43]",
        time_horizon="Medium to Long (2-5 yrs)",
        confidence="Medium",
        rule_id="R10_high_aridity",
    )


def _rule_11(s: SiteState) -> bool:
    return (s.grazing_type == "continuous" and s.land_use in ("pasture", "grassland")
            and s.rainfall_bucket() is not None)

def _build_11(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Switch from continuous grazing to rotational grazing (paddock rest periods of "
                    "roughly 4-6 up to 80-120 days between grazing bouts)",
        impacted_metrics=[
            "Soil organic carbon (+0.1-1.5 tC/ha/yr depending on site and rest regime)",
            "Aboveground plant diversity and productivity",
            "Soil erosion (reduced)",
        ],
        scientific_reasoning=(
            "Resting paddocks between grazing bouts lets plants stay in a more favorable growth "
            "stage and lets roots replenish carbohydrate reserves rather than being persistently "
            "defoliated, which improves organic matter allocation to soil, raises microbial "
            "activity, and reduces the compaction that continuous stocking causes."
        ),
        evidence="FAO 'Recarbonizing Global Soils' Vol.4, Case Study 9 (NSW, Australia): rotational "
                 "grazing raised 0-30cm SOC stocks by 0.13-1.46 tC/ha/yr versus continuous grazing; "
                 "comparable rotational-grazing sequestration averaged ~0.21 tC/ha/yr in Africa and "
                 "~0.69 tC/ha/yr in South America in the same synthesis",
        time_horizon="Medium (4-10 yrs, per the cited NSW sites)",
        confidence="Medium",
        rule_id="R11_continuous_grazing",
    )


def _rule_12(s: SiteState) -> bool:
    return (s.biodiversity_habitat_index is not None and s.biodiversity_habitat_index < 0.55
            and s.land_use is not None)

def _build_12(s: SiteState) -> Recommendation:
    return Recommendation(
        what_to_do="Treat this site as a habitat-restoration priority: combine hedgerow/live-fence "
                    "corridors with reduced agrochemical intensity to raise the local habitat "
                    "condition score",
        impacted_metrics=[
            "Biodiversity Habitat Index (currently below the 0.55 threshold used here to flag "
            "degraded landscapes in this dataset)",
            "Species richness",
            "Soil organic carbon (secondary benefit of woody corridors)",
        ],
        scientific_reasoning=(
            "A low, and often still-declining, remotely-sensed habitat-condition index indicates "
            "the surrounding landscape has lost structural and species diversity relative to "
            "intact reference habitat (e.g. nearby forest). Woody corridors and live fences "
            "directly add the structural habitat elements that raise this class of index, and have "
            "the secondary benefit of raising soil organic carbon."
        ),
        evidence="CSIRO Biodiversity Habitat Index v4 (India, BILBI model), geo-lookup for this "
                 "site's coordinates; live fences in a comparable degraded-pasture landscape raised "
                 "SOC by up to 1.2 tC/ha/yr within 9 years (FAO Recarbonizing Global Soils Vol.4, "
                 "Case Study 36)",
        time_horizon="Medium to Long (site-level habitat recovery is typically 5-10+ yrs)",
        confidence="Medium",
        rule_id="R12_low_bhi",
    )


RULES = [
    (_rule_1, _build_1),
    (_rule_2, _build_2),
    (_rule_3, _build_3),
    (_rule_4, _build_4),
    (_rule_5, _build_5),
    (_rule_6, _build_6),
    (_rule_7, _build_7),
    (_rule_8, _build_8),
    (_rule_9, _build_9),
    (_rule_10, _build_10),
    (_rule_11, _build_11),
    (_rule_12, _build_12),
]

# A minimum viable set of variables needed before we'll attempt ANY reasoning,
# used by conversation_agent.py to decide whether to ask clarifying questions.
CORE_VARS_FOR_DIAGNOSIS = ["soc_percent", "rainfall_bucket", "land_use"]


def missing_core_vars(s: SiteState) -> List[str]:
    missing = []
    if s.soc_percent is None:
        missing.append("soil organic carbon % (current estimate)")
    if s.rainfall_bucket() is None:
        missing.append("annual rainfall pattern (low/medium/high, or mm/yr)")
    if s.land_use is None:
        missing.append("current land use (monoculture, mixed cropping, agroforestry, pasture, fallow)")
    return missing


def run_reasoning(s: SiteState) -> List[Recommendation]:
    """Fire every rule whose condition matches the given site state."""
    recs = []
    for condition, builder in RULES:
        try:
            if condition(s):
                recs.append(builder(s))
        except Exception:
            continue
    return recs
