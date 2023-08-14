relation_classes = {
    "per:charges": ["per:sentence", "per:allegations", "per:indictment", "per:offense", "per:conviction"],
    "per:date_of_death": ["per:age", "per:country_of_birth", "per:origin", "per:cause_of_death", "per:city_of_death"],
    "per:country_of_death": ["per:city_of_death", "per:stateorprovince_of_death", "per:origin", "per:date_of_death", "per:stateorprovince_of_birth"],
    "per:cause_of_death": ["per:age", "per:origin", "per:date_of_birth", "per:date_of_death", "per:city_of_death"],
    "org:founded_by": ["org:shareholders", "per:other_family", "org:members", "org:political/religious_affiliation", "org:number_of_employees/members"],
    "org:founded": ["org:dissolved", "org:stateorprovince_of_branch", "org:country_of_branch", "org:city_of_branch", "org:website"],
    "per:city_of_death": ["per:city_of_birth", "per:stateorprovince_of_death", "per:country_of_death", "per:stateorprovince_of_birth", "per:origin"],
    "per:stateorprovince_of_death": ["per:stateorprovince_of_birth", "per:origin", "per:city_of_death", "per:city_of_birth", "per:country_of_death"],
    "per:date_of_birth": ["per:age", "per:country_of_birth", "per:stateorprovinces_of_residence", "per:cities_of_residence", "org:members"],
    "per:stateorprovince_of_birth": ["per:stateorprovince_of_death", "per:country_of_birth", "per:date_of_birth", "per:city_of_birth", "per:origin"],
    "per:country_of_birth": ["per:country_of_death", "per:origin", "per:city_of_birth", "per:stateorprovince_of_birth", "per:stateorprovinces_of_residence"],
    "per:city_of_birth": ["per:city_of_death", "per:stateorprovince_of_birth", "per:country_of_birth", "per:country_of_death", "per:origin"],
    "org:shareholders": ["org:members", "org:political/religious_affiliation", "org:founded_by", "org:website", "org:alternate_names"],
    "per:other_family": ["org:members", "org:founded_by", "per:siblings", "org:shareholders", "org:political/religious_affiliation"],
    "per:title": ["per:employee_of", "org:members", "per:other_family", "org:top_members/employees", "org:political/religious_affiliation"],
    "org:dissolved": ["org:founded", "org:website", "org:city_of_branch", "org:stateorprovince_of_branch", "org:country_of_branch"],
    "per:countries_of_residence": ["per:stateorprovinces_of_residence", "per:cities_of_residence", "org:members", "org:top_members/employees", "per:origin"],
    "per:stateorprovinces_of_residence": ["per:countries_of_residence", "per:cities_of_residence", "per:origin", "org:members", "org:top_members/employees"],
    "per:cities_of_residence": ["per:countries_of_residence", "per:stateorprovinces_of_residence", "per:origin", "org:members", "org:top_members/employees"],
    "org:member_of": ["org:political/religious_affiliation", "org:website", "org:shareholders", "org:members", "org:founded_by"],
    "per:religion": ["org:political/religious_affiliation", "org:members", "org:website", "org:founded_by", "per:origin"],
    "org:political/religious_affiliation": ["org:member_of", "org:website", "org:shareholders", "org:members", "org:alternate_names"],
    "org:top_members/employees": ["per:employee_of", "per:title", "per:other_family", "org:members", "org:political/religious_affiliation"],
    "org:number_of_employees/members": ["org:members", "org:top_members/employees", "org:website", "org:founded_by", "org:political/religious_affiliation"],
    "per:schools_attended": ["org:member_of", "org:political/religious_affiliation", "org:website", "org:shareholders", "org:founded_by"],
    "per:employee_of": ["per:title", "per:other_family", "org:members", "org:top_members/employees", "org:political/religious_affiliation"],
    "per:siblings": ["org:members", "org:shareholders", "org:political/religious_affiliation", "per:other_family", "per:spouse"],
    "per:spouse": ["per:other_family", "org:members", "org:shareholders", "org:political/religious_affiliation", "per:siblings"],
    "per:parents": ["per:other_family", "org:members", "org:shareholders", "org:political/religious_affiliation", "per:children"],
    "per:children": ["per:other_family", "org:members", "org:shareholders", "org:political/religious_affiliation", "per:parents"],
    "org:alternate_names": ["org:members", "org:website", "org:founded_by", "org:political/religious_affiliation", "org:shareholders"],
    "org:members": ["org:website", "org:shareholders", "org:founded_by", "org:political/religious_affiliation", "org:alternate_names"],
    "per:origin": ["per:religion", "org:political/religious_affiliation", "per:identity", "org:shareholders", "org:members"],
    "org:website": ["org:members", "org:founded_by", "org:political/religious_affiliation", "org:shareholders", "org:alternate_names"],
    "per:age": ["per:countries_of_residence", "per:stateorprovinces_of_residence", "per:cities_of_residence", "org:members", "org:political/religious_affiliation"],
    "no_relation": ["per:sentence", "per:cause_of_death", "per:date_of_death", "org:founded", "org:stateorprovince_of_branch", "per:charges"],
    "per:identity": ["per:title", "org:alternate_names", "org:members", "org:political/religious_affiliation", "per:schools_attended"],
    "org:stateorprovince_of_branch": ["org:country_of_branch", "org:city_of_branch", "org:website", "org:dissolved", "org:founded"],
    "org:country_of_branch": ["org:city_of_branch", "org:stateorprovince_of_branch", "org:website", "org:dissolved", "org:founded"],
    "org:city_of_branch": ["org:country_of_branch", "org:stateorprovince_of_branch", "org:website", "org:dissolved", "org:founded"]
}


import random

def modify_relation_classes(relation_classes):
    modified_classes = relation_classes.copy()
    
    for key, entities in modified_classes.items():
        if key == "no_relation":
            continue
        
        if random.random() < 0.7:
            random_index = random.randint(0, len(entities) - 1)
            entities[random_index] = "no_relation"
    
    return modified_classes

# Test the function
modified_relation_classes = modify_relation_classes(relation_classes)

# Print the modified relation classes
for key, entities in modified_relation_classes.items():
    print(key, entities)
