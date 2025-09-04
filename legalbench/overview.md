# LegalBench Global Tasks Overview

This document provides an overview of LegalBench tasks that have `automatic_evaluation` set to "true" and `jurisdiction` set to "global". These tasks represent legal reasoning challenges that are applicable across multiple jurisdictions rather than being specific to a particular country's legal system.

## Summary Statistics

- **Total Global Tasks with Automatic Evaluation**: 28 tasks
- **Main Categories**: Contract Analysis, Legal Reasoning, Privacy Policy Analysis, Evidence Law, and Judicial Interpretation

## Task Categories and Topics

### 1. Contract Understanding and Analysis (18 tasks)

**CUAD (Contract Understanding Atticus Dataset) Tasks (10 tasks)**
These tasks focus on identifying specific types of clauses in contracts:

- **License and IP Management**:
  - `cuad_affiliate_license-licensee` & `cuad_affiliate_license-licensor`: Identify affiliate licensing clauses
  - `cuad_ip_ownership_assignment`: Classify IP ownership assignment clauses
  - `cuad_irrevocable_or_perpetual_license`: Identify irrevocable/perpetual license terms
  - `cuad_license_grant`: Classify license grant clauses
  - `cuad_unlimited-all-you-can-eat-license`: Identify unlimited license provisions

- **Legal and Financial Protection**:
  - `cuad_cap_on_liability`: Identify liability caps
  - `cuad_covenant_not_to_sue`: Classify covenant not to sue clauses
  - `cuad_governing_law`: Identify governing law provisions
  - `cuad_revenue-profit_sharing`: Classify revenue/profit sharing terms

**MAUD (Merger Agreement Understanding Dataset) Tasks (8 tasks)**
These tasks focus on merger and acquisition agreement analysis:

- **Material Adverse Effect (MAE) Analysis**:
  - `maud_ability_to_consummate_concept_is_subject_to_mae_carveouts`: Analyze MAE carveouts
  - `maud_change_in_law__subject_to_disproportionate_impact_modifier`: Legal change impact analysis
  - `maud_changes_in_gaap_or_other_accounting_principles__subject_to_disproportionate_impact_modifier`: Accounting changes analysis
  - `maud_fls_(mae)_standard`: MAE standard classification
  - `maud_general_economic_and_financial_conditions_subject_to_disproportionate_impact_modifier`: Economic conditions analysis
  - `maud_pandemic_or_other_public_health_event__subject_to_disproportionate_impact_modifier`: Pandemic impact analysis
  - `maud_pandemic_or_other_public_health_event_specific_reference_to_pandemic-related_governmental_responses_or_measures`: COVID-specific provisions
  - `maud_relational_language_(mae)_applies_to`: MAE relational language analysis

**Use Cases**: 
- Legal document review automation
- Contract compliance checking
- Due diligence processes
- Risk assessment in M&A transactions
- Legal AI assistants for contract analysis

### 2. Privacy Policy Analysis (3 tasks)

- `privacy_policy_entailment`: Verify if descriptions of privacy policy clauses are accurate
- `opp115_do_not_track`: Identify Do Not Track signal handling clauses
- `opp115_international_and_specific_audiences`: Classify international audience provisions

**Use Cases**:
- Privacy compliance auditing
- GDPR/CCPA compliance checking
- Privacy policy generation and review
- Consumer rights protection
- Legal tech for privacy law

### 3. Legal Text Analysis and Interpretation (4 tasks)

- `definition_classification`: Identify sentences that define legal terms
- `definition_extraction`: Extract specific terms being defined in legal text
- `overruling`: Classify whether judicial sentences overrule previous cases
- `textualism_tool_dictionaries`: Identify dictionary-based statutory interpretation

**Use Cases**:
- Legal research and case law analysis
- Judicial opinion summarization
- Legal precedent tracking
- Statutory interpretation assistance
- Legal education tools

### 4. Evidence Law (1 task)

- `hearsay`: Classify evidence as hearsay or admissible

**Use Cases**:
- Trial preparation assistance
- Legal education and training
- Evidence evaluation tools
- Litigation support systems

### 5. Civil Litigation (1 task)

- `ssla_plaintiff`: Extract plaintiff identities from securities class action complaints

**Use Cases**:
- Case management systems
- Legal document processing
- Litigation analytics
- Class action administration

### 6. International Law (1 task)

- `international_citizenship_questions`: Answer questions about citizenship law worldwide

**Use Cases**:
- Immigration law practice
- Citizenship consulting
- International legal services
- Government policy analysis

## Key Applications and Use Cases

### Legal Technology Platforms
These tasks are ideal for developing:
- **Contract Analysis Tools**: Automated contract review, clause identification, and risk assessment
- **Legal Research Platforms**: Case law analysis, precedent identification, and judicial opinion processing
- **Compliance Systems**: Privacy policy auditing, regulatory compliance checking

### Professional Legal Services
- **Law Firms**: Document review automation, due diligence support, contract analysis
- **Corporate Legal Departments**: Contract management, compliance monitoring, risk assessment
- **Legal Consultancies**: International law advice, privacy compliance, litigation support

### Educational Applications
- **Law Schools**: Teaching tools for contract law, evidence law, and legal reasoning
- **Legal Training**: Professional development programs for lawyers and paralegals
- **Legal Research**: Academic research in legal NLP and AI applications

### Regulatory and Government Use
- **Regulatory Bodies**: Compliance monitoring, policy analysis
- **Courts**: Case management, document processing
- **Government Agencies**: Legal document analysis, policy development

## Technical Characteristics

All tasks feature:
- **Automatic Evaluation**: Standardized metrics for consistent performance assessment
- **Global Jurisdiction**: Applicable across multiple legal systems
- **Diverse Task Types**: Classification, extraction, and entailment tasks
- **Real-World Data**: Based on actual legal documents and court opinions
- **Scalable Applications**: Suitable for both academic research and commercial deployment

## Conclusion

These 28 global LegalBench tasks represent a comprehensive suite of legal reasoning challenges that transcend specific jurisdictions. They cover fundamental legal concepts including contract interpretation, privacy law, evidence rules, and judicial reasoning that are applicable across different legal systems. The tasks provide excellent benchmarks for developing legal AI systems with broad applicability and real-world utility.